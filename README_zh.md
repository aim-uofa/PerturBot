<div align="center">

<h1>PerturBot: Breaking Shortcut Priors in<br>Vision-Language-Action Models with Perturbative Training</h1>

<p>
  <b>Mingyu Liu</b><sup>1,2,*</sup> · <b>Chonghao Sima</b><sup>3,*</sup> · <b>Tianjian Feng</b><sup>1</sup> · <b>Hanqing Wang</b><sup>4</sup><br>
  <b>Cong Chen</b><sup>1</sup> · <b>Hao Chen</b><sup>1,†</sup> · <b>Chunhua Shen</b><sup>1,†</sup>
</p>

<p>
  <sup>1</sup> 浙江大学 &nbsp; <sup>2</sup> 上海创智学院<br>
  <sup>3</sup> 香港大学 &nbsp; <sup>4</sup> 香港科技大学（广州）<br>
  <sup>*</sup> 共同一作 &nbsp; <sup>†</sup> 通讯作者
</p>

<p><a href="https://openreview.net/forum?id=r7zfsysr22">📄 论文（OpenReview）</a> &nbsp;·&nbsp; <a href="#快速开始">🚀 快速开始</a> &nbsp;·&nbsp; <a href="#引用">引用</a> &nbsp;·&nbsp; <a href="README.md">English</a></p>

</div>

## 论文简介

机器人能完成熟悉的任务，不代表它使用了正确的决策依据：显眼物体可能盖过指令中的目标，熟悉名词可能触发错误操作，夹爪抓空后也可能继续抬起。**PerturBot** 针对这些“模态捷径”改变训练数据，让策略更多依赖任务证据，而不增加推理时的额外模块。

论文包含三类数据干预，以及一个衡量证据使用情况的诊断指标：

- **V — 腕部视角扰动：** 改变无关视觉线索，同时保持任务及动作监督有效。
- **C — 指令细化：** 在不违背已记录动作的前提下，明确对象、操作及相关状态。
- **R — 轨迹扩展：** 筛选随机运动与失败执行中的片段，按片段实际行为重新标注。
- **GroundFscore：** 通过成对输入编辑，区分面对无关变化时的稳定性和面对任务相关证据时的响应能力，与任务成功率互补。

<p align="center">
  <a href="docs/assets/overview.svg"><img src="docs/assets/overview.svg" width="100%" alt="PerturBot 主图：视觉显著性、名词和运动惯性造成的模态捷径，以及腕部视角扰动、指令细化和轨迹扩展三类训练干预。"></a>
</p>

论文在真实机器人 General Pick and Place 与 RoboTwin 2.0 上评测这一思路。主图展示捷径行为与三类训练干预；推理流程保持不变。

> **当前代码范围：** 本仓库提供基于 [openpi](https://github.com/Physical-Intelligence/openpi) 的 PiperX π₀.₅ 训练与推理骨干，以及数据转换、归一化和 checkpoint 工具。论文专用的 V/C/R 数据构造流程和 GroundFscore 评测器不在本次源码快照中；训练数据和微调权重也未包含。

## 快速开始

**环境：** Linux x86-64、Python 3.11、[uv](https://docs.astral.sh/uv/) 和兼容 CUDA 12 的驱动。训练使用**单机 8 卡**，建议 A100/H100 80 GB 级别硬件。

### 1. 安装

```bash
git clone https://github.com/aim-uofa/PerturBot.git
cd PerturBot
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
export HF_LEROBOT_HOME="$PWD/data/lerobot"
```

### 2. 准备数据并训练

按[数据格式](docs/training.md#2-data-contract)准备 episode manifest；已有 `local/piperx` 的兼容 LeRobot v2.1 数据时可跳过转换。

```bash
uv run examples/piperx_real/convert_data.py \
  --manifest /path/to/episodes.json --raw-root /path/to/raw_data \
  --repo-id local/piperx
JAX_PLATFORMS=cpu uv run scripts/compute_norm_stats.py \
  --config-name pi05_piperx --repo-id local/piperx
bash scripts/train_8gpu.sh --exp-name piperx_run --data.repo-id local/piperx
```

启动器用一个 JAX 进程使用 8 张 GPU，不需要 `torchrun`。训练加载公开 π₀.₅ 基座，checkpoint 保存到 `checkpoints/pi05_piperx/piperx_run/`。

### 3. 推理

```bash
CUDA_VISIBLE_DEVICES=0 uv run examples/piperx_real/infer.py \
  --checkpoint checkpoints/pi05_piperx/piperx_run/15000 \
  --dummy --prompt 'Place the object on the plate.'
```

`--dummy` 仅检查接口，不操作机器人。真实观测用 `--observation observation.npz` 替代；归一化统计和动作约定必须与 checkpoint 匹配，接入硬件前阅读[输入与安全约定](docs/inference.md)。

**详细说明：** [训练 / 续训](docs/training.md) · [推理 / WebSocket](docs/inference.md) · [Checkpoint 选择](docs/checkpoints.md) · [测试与验证边界](docs/validation.md)

## 引用

```bibtex
@misc{liu2026perturbot,
  title={PerturBot: Breaking Shortcut Priors in Vision-Language-Action Models with Perturbative Training},
  author={Mingyu Liu and Chonghao Sima and Tianjian Feng and Hanqing Wang and Cong Chen and Hao Chen and Chunhua Shen},
  year={2026},
  url={https://openreview.net/forum?id=r7zfsysr22}
}
```

## 致谢与许可

本项目基于 [Physical Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，保留上游 [Apache 2.0 许可证](LICENSE)、[Gemma 条款](LICENSE_GEMMA.txt)和 [NOTICE](NOTICE)。本仓库不是 Physical Intelligence 的官方发布；数据与模型权重需要单独获得使用和发布许可。
