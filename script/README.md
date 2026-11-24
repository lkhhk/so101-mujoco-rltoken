# 实验文件与运行说明

## 实验文件

- [config/](../config/) 存放实验参数、模型校验值、种子范围和 SO-101 仿真关节映射。
- [results/](../results/) 存放训练批次统计、50 个种子的逐回合评测和两组配对基线。
- [script/](./) 存放本次续训、评测及所用的仿真和控制代码。

## 运行训练和评测

使用 Linux、CUDA 和 Python 3.11。版本记录见 [实验配置](../config/warmup20-episode1024.json)。运行时需要 LeRobot 0.4.4 的 π₀.₅ 推理实现以及 [RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open/tree/e3c1f4517108fdff66d93b68bae3a16fc1cd5dd3) 的 `pi05_rlt` 模块。可以把该提交的 `lerobot/src/lerobot/policies/pi05_rlt` 目录放在 `script/rlt-stage1-b16-b1-20261122/upstream/pi05_rlt`，训练脚本会将该目录的父目录加入 LeRobot 的策略搜索路径。

准备以下文件。路径相对于仓库根目录，可以用软链接指向现有文件。请用实验配置中的 SHA-256 核对模型权重。

| 文件 | 放置路径 |
|---|---|
| IRA π₀.₅ 008000 底座 | `ira-pi05-v7-full-v2-008000-blockbox-20261117/` |
| 累计 30,000 步的仿真 SFT LoRA，含配置和处理器文件 | `script/grpo-000140-sft-mixed77-v1/checkpoints/010000/` |
| PaliGemma tokenizer | `paligemma-tokenizer-35e4f464/` |
| SO-101 MuJoCo Menagerie 模型 XML 和网格 | `script/menagerie/so101.xml` 和 `script/menagerie/assets/` |
| 累计 20,512 步的 RL Token encoder | `script/rlt-stage1-30000-b64-20261123/additional-20000-total-20512.pt` |
| 续训起点 | `script/rlt-stage2-fresh-encoder20k-1024episodes-b1-20261123/episode-0640.pt` |
| 最终检查点，从本仓库 Release 下载 | `script/rlt-stage2-new-encoder-640-warmup20-b1-20261124/episode-1024.pt` |

从仓库根目录运行。评测只需要最终检查点，续训需要第 640 回合起点。

```bash
export PYTHONPATH="$PWD/script${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1

python script/rlt-stage2-new-encoder-640-warmup20-b1-20261124/continue_warmup20.py
python script/rlt-stage2-eval-warmup20-1024-paired50-20261124/evaluate_1024.py
```

续训脚本在自身目录保存检查点和每 100 个正式训练回合的入盒率。评测脚本在自身目录写入新的 `report.json`，并与 `results/paired-baselines.json` 按种子配对。
