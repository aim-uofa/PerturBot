"""Exercise real JAX/FSDP/Orbax plumbing with a small synthetic model, not pi0.5 weights."""

import dataclasses
import json

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.training import config as _config
from openpi.training import data_loader
from scripts import train


class TinyModel(_model.BaseModel):
    def __init__(self, config, rng):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        rngs = nnx.Rngs(rng)
        self.input_layer = nnx.Linear(config.action_dim, 1024, rngs=rngs)
        # Exactly 4 MiB: large enough to exercise the production FSDP sharding threshold.
        self.hidden = nnx.Linear(1024, 1024, rngs=rngs)
        self.output_layer = nnx.Linear(1024, config.action_dim, rngs=rngs)

    @override
    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        values = self.output_layer(self.hidden(self.input_layer(observation.state)))
        return jnp.broadcast_to(values[:, None, :], (values.shape[0], self.action_horizon, self.action_dim))

    @override
    def compute_loss(self, rng, observation, actions, *, train=False):
        del train
        return jnp.mean(jnp.square(self.sample_actions(rng, observation) - actions), axis=-1)


@dataclasses.dataclass(frozen=True)
class TinyConfig(_model.BaseModelConfig):
    action_dim: int = 16
    action_horizon: int = 2
    max_token_len: int = 4

    @property
    @override
    def model_type(self):
        return _model.ModelType.PI0

    @override
    def create(self, rng):
        return TinyModel(self, rng)

    @override
    def inputs_spec(self, *, batch_size=1):
        with at.disable_typechecking():
            observation = _model.Observation(
                images={"base_0_rgb": jax.ShapeDtypeStruct((batch_size, 16, 16, 3), jnp.float32)},
                image_masks={"base_0_rgb": jax.ShapeDtypeStruct((batch_size,), jnp.bool_)},
                state=jax.ShapeDtypeStruct((batch_size, self.action_dim), jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct((batch_size, self.max_token_len), jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct((batch_size, self.max_token_len), jnp.bool_),
            )
        actions = jax.ShapeDtypeStruct((batch_size, self.action_horizon, self.action_dim), jnp.float32)
        return observation, actions


def test_train_resume_and_parameter_restore(tmp_path, monkeypatch):
    # Isolate validation-loop plumbing from dataset downloads. Split correctness
    # and normalization/episode filtering are tested independently in test_piperx.
    monkeypatch.setattr(data_loader, "compute_train_val_episodes", lambda *args, **kwargs: ([0, 1], [2]))
    devices = jax.device_count()
    config = _config.TrainConfig(
        name="tiny",
        exp_name="smoke",
        model=TinyConfig(),
        data=_config.FakeDataConfig(),
        batch_size=devices * 2,
        fsdp_devices=devices,
        checkpoint_base_dir=str(tmp_path),
        num_workers=0,
        num_train_steps=2,
        log_interval=1,
        save_interval=2,
        keep_period=2,
        val_episodes_frac=0.2,
        val_interval=2,
        val_batches=1,
        val_action_dim=14,
        wandb_enabled=False,
    )
    train.main(config)
    assert (config.checkpoint_dir / "2" / "params" / "_METADATA").is_file()
    train.main(dataclasses.replace(config, resume=True, num_train_steps=4))
    assert (config.checkpoint_dir / "4" / "params" / "_METADATA").is_file()
    metrics = [json.loads(line) for line in (config.checkpoint_dir / "metrics.jsonl").read_text().splitlines()]
    assert [entry["step"] for entry in metrics] == [1, 2, 3, 4]
    assert all("val/action_mse" in entry for entry in metrics if entry["step"] % 2 == 0)
    params = _model.restore_params(config.checkpoint_dir / "4" / "params")
    restored = config.model.load(params)
    actions = restored.sample_actions(jax.random.key(0), config.model.fake_obs())
    assert actions.shape == (1, 2, 16)
    assert np.isfinite(np.asarray(actions)).all()
