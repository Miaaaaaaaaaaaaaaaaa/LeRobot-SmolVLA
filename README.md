

# YeahBot-C1 × LeRobot：SmolVLA 训练流程

以 **LeRobot** 为框架，使用 **SmolVLA** 在 **yeahbot-C1** 数据上训练。
原始数据在 `local_data_raw`。

---

## 1. 数据格式说明

- **原始数据**：`local_data_raw/` 下每个 `session_*` 文件夹为一轮示教（一个 episode）。
- 每个 session 需包含：
  - `camera/image_index.csv`：时间戳 → 图像文件名
  - `camera/image_*.jpg`：图像
  - `joint_states/joint_states_data.csv`：时间戳与关节位置（分号分隔的 15 维：4 轮 + 11 臂）
- 转换脚本会按时间戳将图像与关节状态对齐，**action** 取为下一帧的关节状态（模仿学习）。

---

## 2. 在 AutoDL 上的推荐流程

### 2.1 环境准备

```bash
# 创建环境（示例 Python 3.10）
conda create -n lerobot python=3.10 -y
conda activate lerobot

# 安装 LeRobot
# 仅 SmolVLA
pip install "lerobot[smolvla]"

# 或从源码安装（支持最新 v3 数据集）
pip install git+https://github.com/huggingface/lerobot.git
pip install ".[smolvla]"   # 在 lerobot 源码目录下
```

若使用本仓库提供的依赖列表（可选）：

```bash
pip install -r requirements.txt
```

若安装后出现 **numba 与 numpy 版本冲突**（例如 `numba 0.59.1 requires numpy<1.27, but you have numpy 2.x`），可升级 numba 以兼容 numpy 2.x：

```bash
pip install "numba>=0.63"
```

### 2.2 数据转换：原始数据 → LeRobot v3

将 `local_data_raw` 转为 LeRobot 数据集（默认输出到 `./lerobot_yeahbot_c1`）：

```bash
# 仅本地输出
python scripts/convert_yeahbot_to_lerobot.py \
  --raw-dir local_data_raw \
  --output-dir ./lerobot_yeahbot_c1

# 转换后并上传到 Hugging Face Hub（需先 huggingface-cli login）
python scripts/convert_yeahbot_to_lerobot.py \
  --raw-dir local_data_raw \
  --output-dir ./lerobot_yeahbot_c1 \
  --push-to-hub \
  --repo-id YOUR_HF_USER/yeahbot-c1
```

参数说明：

- `--raw-dir`：原始数据根目录（包含多个 `session_*`）。
- `--output-dir`：LeRobot 数据集输出目录（若已存在会先删除再创建）。
- `--repo-id`：Hub 上的数据集 ID，上传时必填。
- `--fps`：数据集 FPS（默认 10）。
- `--state-dim`：状态/动作维度（默认 15）。

### 2.3 训练 SmolVLA

#### 2.3.1 基础训练命令（本地数据集）
```bash
# 设置离线模式（避免访问 HF Hub 下载预训练权重，若已下载）
export HF_HUB_OFFLINE=1

# 核心训练命令
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --rename_map='{"observation.images.front": "observation.images.camera1"}' \
    --dataset.repo_id=smolvla_dataset\
    --dataset.root=/root/autodl-tmp/datasets/smolvla_dataset\
    --batch_size=32 \
    --num_workers=6 \
    --steps=20000 \
    --output_dir=outputs/train/yeahbot_smolvla \
    --policy.device=cuda \
    --save_checkpoint=true \
    --save_freq=1000 \
    --optimizer.type=adamw \
    --optimizer.lr=1e-5 \
    --optimizer.weight_decay=0.01 \
    --resume=false
```

#### 2.3.2 简化训练脚本调用（兼容原有方式）
```bash
# 使用本地转换好的数据集（在项目根目录执行）
export DATASET_REPO_ID=./lerobot_yeahbot_c1
bash scripts/train_smolvla.sh

# 或直接指定步数、batch、输出目录
STEPS=20000 BATCH_SIZE=32 OUTPUT_DIR=outputs/my_smolvla bash scripts/train_smolvla.sh
```

若数据集已上传到 Hub：
```bash
export DATASET_REPO_ID=YOUR_HF_USER/yeahbot-c1
bash scripts/train_smolvla.sh
```

#### 2.3.3 关键训练参数说明
| 参数 | 说明 |
|------|------|
| `HF_HUB_OFFLINE=1` | 离线模式，避免重复下载预训练权重（AutoDL 环境推荐） |
| `--rename_map` | 映射数据集图像字段名到模型期望的字段名（解决 camera1/front 命名不一致） |
| `--dataset.root` | AutoDL 中数据集的实际存储路径（需替换为你的路径） |
| `--optimizer.lr=1e-5` | 针对 15 维关节数据调低学习率，避免前期 loss 过大 |
| `--save_freq=1000` | 每 1000 step 保存一次 checkpoint，便于中断后恢复训练 |
| `--num_workers=6` | 数据加载线程数，适配 AutoDL 实例的 CPU 核心数 |

---

## 3. 脚本与文件一览

| 文件 | 说明 |
|------|------|
| `scripts/convert_yeahbot_to_lerobot.py` | 原始数据 → LeRobot v3，按 session 为 episode，图像与 joint 按时间戳对齐 |
| `scripts/train_smolvla.sh` | SmolVLA 训练入口，支持本地/Hub 数据集 |
| `requirements_autodl.txt` | 可选依赖列表（含 lerobot 源码安装方式） |

---

## 4. 注意事项

1. **LeRobot 版本**：数据集为 v3 格式，需使用支持 v3 的 LeRobot（如从 main 安装或 ≥0.4）。若 `LeRobotDataset.create` / `add_frame` 报错，请升级或从 GitHub 安装。
2. **显存**：SmolVLA 可用较小 batch（如 32）；π₀-FAST 建议开启 `gradient_checkpointing`，batch 约 4。
3. **任务描述**：转换时每个 episode 的 task 来自 `metadata.json` 的 `robot` 或默认「完成 YeahBot 示教任务」。训练/评估时使用的语言指令需与数据集中任务描述一致。
4. **AutoDL**：数据若在云盘，先将 `local_data_raw` 和（或）`lerobot_yeahbot_c1` 放到实例可访问路径，再设置 `DATASET_REPO_ID` 为对应路径或 Hub repo_id。
5. **路径替换**：训练命令中 `/root/autodl-tmp/datasets/smolvla_dataset` 需替换为你在 AutoDL 上的实际数据集路径。

---

## 5. 训练 loss 过大时的排查（如 step 200/400 时 loss ≈ 1e6）

可能原因与应对：

| 原因 | 说明 | 建议 |
|------|------|------|
| **State/Action 维度与预训练不一致** | `lerobot/smolvla_base` 默认按 6 维 state/action（如 PushT）预训练，YeahBot 为 **15 维**。策略会按数据集扩展维度，但新增维度的头是随机初始化，前期误差会很大。 | 正常现象。可先多跑几步（如 2k–5k）看 loss 是否持续下降；若一直不降再考虑下面几条。 |
| **归一化与数值尺度** | 数据集中 state/action 为原始关节角（未做全局归一化），策略内部用 MEAN_STD 归一化。若某维方差很小或存在异常值，归一化后目标尺度可能很大，MSE 会被放大。 | 转换脚本已对 `episode_stats` 的 **std 设下限 1e-2**，避免某维几乎不动时 `(x-mean)/std` 爆炸导致 loss/grdn 极大。若数据集是旧版转换的，需**重新运行转换**再训练。 |
| **学习率偏大** | 15 维新头 + 较大 lr 时，前期梯度大，loss 会很高。 | 训练命令中已将 lr 设为 `1e-5`（默认通常 3e-5），若仍大可再降至 `5e-6`。 |
| **Flow matching 目标尺度** | SmolVLA 用 flow matching；loss 是流匹配目标，与 MSE 不同，绝对值可能较大。只要 **随 step 单调下降** 即表示在学。 | 重点看 loss 曲线趋势，不必纠结绝对数值；若几千 step 后仍不降再排查维度和归一化。 |

**建议操作**：先跑满 2k–5k step 观察曲线。若 loss 持续下降（哪怕缓慢），可继续训练；若基本不降，再尝试 `--optimizer.lr=5e-5` 或检查数据集 meta 中的 state/action 统计量。

---

## 6. 参考链接

- [LeRobot 安装与数据集 v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [SmolVLA 训练](https://huggingface.co/docs/lerobot/smolvla)



