import dataclasses
import functools
import json
import logging
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.provenance as provenance
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def val_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """Validation step: action MSE between sample_actions and ground-truth actions.

    Uses EMA params if available (closer to deployed policy). Runs flow-matching
    integration with `config.val_num_diffusion_steps` steps. Computes MSE in the
    normalized action space (same space as compute_loss target), averaged over
    batch / chunk_horizon / action_dim.
    """
    eval_params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, eval_params)
    model.eval()

    observation, actions = batch
    pred_actions = model.sample_actions(rng, observation, num_steps=config.val_num_diffusion_steps)
    action_dim = config.val_action_dim or actions.shape[-1]
    return {"action_mse": jnp.mean(jnp.square(pred_actions[..., :action_dim] - actions[..., :action_dim]))}


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info("JAX devices: %s", jax.device_count())

    n_dev = jax.device_count()
    if config.batch_size % n_dev != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices ({n_dev}). "
            "Increase --batch-size to a multiple of that count, or use CUDA_VISIBLE_DEVICES=0 for single-GPU."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    provenance.validate_norm_stats(config)

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    train_episodes, val_episodes = _data_loader.compute_train_val_episodes(
        config.data.repo_id, config.val_episodes_frac, seed=config.seed
    )
    val_enabled = val_episodes is not None and len(val_episodes) > 0
    if val_enabled:
        logging.info(
            f"Train/val split: {len(train_episodes)} train episodes, {len(val_episodes)} val episodes "
            f"(val_frac={config.val_episodes_frac}, val_interval={config.val_interval}, "
            f"val_batches={config.val_batches}, val_num_diffusion_steps={config.val_num_diffusion_steps})"
        )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        episodes=train_episodes,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    val_data_loader = None
    if val_enabled:
        val_data_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            episodes=val_episodes,
        )
        logging.info("Initialized val data loader.")

    # Tracking is opt-in; local metrics are always written below.
    if config.wandb_enabled:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, config.batch_size))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)
    run_info = {
        "config": config.name,
        "repo_id": config.data.repo_id,
        "batch_size": config.batch_size,
        "fsdp_devices": config.fsdp_devices,
        "num_train_steps": config.num_train_steps,
        "seed": config.seed,
        "action_horizon": config.model.action_horizon,
        "extra_delta_transform": getattr(config.data, "extra_delta_transform", None),
        "val_episodes_frac": config.val_episodes_frac,
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
    }
    (config.checkpoint_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    pval_step = None
    if val_enabled:
        pval_step = jax.jit(
            functools.partial(val_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    val_rng = jax.random.fold_in(rng, 1)

    infos = []
    metrics_path = config.checkpoint_dir / "metrics.jsonl"

    def log_metrics(metrics: dict, step: int) -> None:
        record = {"step": step, **{key: float(value) for key, value in metrics.items()}}
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if config.wandb_enabled:
            wandb.log(metrics, step=step)

    try:
        for step in pbar:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            infos.append(info)
            completed_steps = step + 1
            final_step = completed_steps == config.num_train_steps
            iteration_metrics = {}
            if completed_steps % config.log_interval == 0 or final_step:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                info_str = ", ".join(f"{key}={value:.4f}" for key, value in reduced_info.items())
                pbar.write(f"Step {completed_steps}: {info_str}")
                iteration_metrics.update(reduced_info)
                infos = []

            if val_enabled and (completed_steps % config.val_interval == 0 or final_step):
                # Re-evaluate the same batches with the same diffusion noise for comparable metrics.
                val_data_iter = iter(val_data_loader)
                val_infos = []
                for val_index in range(config.val_batches):
                    val_batch = next(val_data_iter)
                    with sharding.set_mesh(mesh):
                        val_infos.append(pval_step(jax.random.fold_in(val_rng, val_index), train_state, val_batch))
                reduced_val = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(val_infos)))
                val_log = {f"val/{key}": value for key, value in reduced_val.items()}
                pbar.write(f"Step {completed_steps} [VAL]: {val_log}")
                iteration_metrics.update(val_log)

            if iteration_metrics:
                log_metrics(iteration_metrics, completed_steps)

            # Directory names and metric steps both denote completed optimizer updates.
            if completed_steps % config.save_interval == 0 or final_step:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, completed_steps)
            if not final_step:
                batch = next(data_iter)
    finally:
        logging.info("Waiting for checkpoint writes to finish")
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()
        wandb.finish()


if __name__ == "__main__":
    main(_config.cli())
