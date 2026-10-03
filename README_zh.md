# openpi / PiperX π₀.₅ 开源整理版

本仓库基于 [Physical Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，提供数据转换、归一化、**单机 8 卡训练**、本地推理和可选 WebSocket 推理。原始模型实现及许可证保留。

源码不包含内部代理、内部服务地址、个人绝对路径、数据集、私有 norm stats、训练日志、微调权重或旧 Git 历史。没有自动上传或自动发布操作。

## 安装与 8 卡训练

建议 Linux、Python 3.11、8×A100-80GB / H100-80GB；显卡数量不等于显存保证。需要兼容 CUDA 12 的 NVIDIA 驱动。首次安装/下载公共基座需要网络，也可以提前准备离线缓存。

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
export HF_LEROBOT_HOME="$PWD/data/lerobot"

# 已有兼容的 LeRobot v2.1 数据时跳过转换。
uv run examples/piperx_real/convert_data.py \
  --manifest /path/to/episodes.json --raw-root /path/to/raw_data \
  --repo-id local/piperx

JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
  --config-name pi05_piperx --repo-id local/piperx

bash scripts/train_8gpu.sh --exp-name piperx_run --data.repo-id local/piperx
```

- 一台机器、**一个 JAX 进程**使用 8 卡，不要再套 `torchrun`，也不需要配置集群通信端口。
- 默认 global batch=256、FSDP=8、15000 步、warmup=1000、学习率=5e-5、EMA=0.999；这是一套起始配方，不保证对任意数据最优。
- 默认不启用 W&B，不需要账号。指标写到 `checkpoints/pi05_piperx/piperx_run/metrics.jsonl`。
- 权重默认从公开 π₀.₅ 基座下载。已有本地权重时追加 `--weight-loader.params-path /path/to/pi05_base/params`。
- 最后一个新 checkpoint 在 `checkpoints/pi05_piperx/piperx_run/15000`，其中包含 `params/`、`assets/`、`train_state/`。
- 续训在原命令追加 `--resume`；不会自动覆盖已有实验。数据 ID、归一化、动作模式和模型结构必须保持一致。
- 默认无验证集。需要验证时，计算 norm stats 和训练**同时**使用相同的 `--val-episodes-frac` 与 `--seed`，见 [训练说明](docs/training.md)。

## 推理

```bash
# 本地调用，不需要任何服务端口。dummy 只测接口，不能控制真实机器人。
CUDA_VISIBLE_DEVICES=0 uv run examples/piperx_real/infer.py \
  --checkpoint checkpoints/pi05_piperx/piperx_run/15000 \
  --dummy --prompt 'Place the object on the plate.'
```

真实输入用 `--observation observation.npz` 替代 `--dummy`。NPZ 中包含三路 RGB `image`、`wrist_left_image`、`wrist_right_image`，以及 float32 的 `state(14,)`。

14 维顺序必须是 **左臂 6 维、右臂 6 维、左夹爪、右夹爪**，不是每只手臂紧接一个夹爪。关节单位保留示教原值，夹爪使用 `range_mm`。输出为 `(10, 14)` 的绝对目标动作块，不是可直接执行的电机指令。

可选远程接口：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --checkpoint checkpoints/pi05_piperx/piperx_run/15000 \
  --host 127.0.0.1 --port 8000
uv run examples/piperx_real/client.py --host 127.0.0.1 --port 8000
```

这是可自行设置的普通推理端口，不是内部端口依赖。服务默认仅监听本机，没有认证/TLS，不要直接暴露公网。具体输入、动作模式和安全注意事项见 [推理说明](docs/inference.md)。

## 旧 checkpoint 特别注意

- `pi05_piperx`：关节 delta、夹爪 absolute；推理时会还原为绝对关节目标。
- `pi05_piperx_absolute`：为旧 absolute-action 训练保留。不能把 absolute checkpoint 用 delta 配置加载，否则关节会被错误地加一次当前 state。
- 保留 `action_horizon=10` 和 `discrete_state_input=False`。后者不同于上游默认值，不应在部署时擅自修改。
- 使用 checkpoint 自带的 norm stats，而不是当前工作区后来重新计算的文件。脚本可识别唯一的旧 asset ID；多个 asset 时需显式指定。
- 训练 loss 最小不代表实机最好。源码包不对历史模型作“最优”结论，也不包含私有评测或权重。

[checkpoint 选择/导出](docs/checkpoints.md) · [发布检查清单](docs/release_checklist.md)

## 已做验证与边界

默认测试 63 项通过；8 个虚拟 CPU 设备上的 FSDP/保存/续训/恢复测试通过。已检查 CLI、锁文件、打包和源码扫描。当前环境没有可见 NVIDIA 设备，**没有完成真实 8 卡或完整 π₀.₅ 权重推理验证**，也没有实机成功率结论。详见 [验证记录](docs/validation.md)。
