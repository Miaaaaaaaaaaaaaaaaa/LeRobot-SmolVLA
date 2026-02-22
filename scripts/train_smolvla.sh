#!/bin/bash
# SmolVLA 训练脚本 - YeahBot-C1 / AutoDL
# 使用前：1) 将原始数据转为 LeRobot 格式（convert_yeahbot_to_lerobot.py）
#        2) 数据集在本地 --dataset.repo_id=./lerobot_yeahbot_c1 或已上传到 Hub --dataset.repo_id=USER/yeahbot-c1
set -e
cd "$(dirname "$0")/.."

# 默认使用 Hugging Face 镜像（国内/AutoDL 等环境 huggingface.co 常不可达）。使用官方站请先 unset HF_ENDPOINT
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 屏蔽 torchvision 视频弃用、torch_dtype 弃用等大量警告；需看警告时在运行前 unset PYTHONWARNINGS
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning,ignore::DeprecationWarning}"

# 可调参数
DATASET_REPO_ID="${DATASET_REPO_ID:-./lerobot_yeahbot_c1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smolvla_yeahbot_c1}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
JOB_NAME="${JOB_NAME:-smolvla_yeahbot_c1}"
# 训练后推送到 Hub 的 repo（设为 local/xxx 则仅本地保存；要推送请设为 USER/repo_name 并启用 push_to_hub）
POLICY_REPO_ID="${POLICY_REPO_ID:-local/smolvla_yeahbot_c1}"
# 未配置 wandb login 时设为 false 可跳过 wandb；需要记录时先 wandb login 再设 WANDB_ENABLE=true
WANDB_ENABLE="${WANDB_ENABLE:-false}"
# RTX 5090 (sm_120) 当前 PyTorch 未支持，会报 "no kernel image available"。可设 USE_CPU=1 用 CPU 训练（很慢），或换 A100/4090 等实例
POLICY_DEVICE="${USE_CPU:+cpu}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
# loss 很大时可尝试降低学习率，例如 OPTIMIZER_LR=5e-5 bash scripts/train_smolvla.sh
OPTIMIZER_LR="${OPTIMIZER_LR:-}"

# 本地数据集：设为 HF_LEROBOT_HOME 下同名目录，便于 lerobot-train 加载（0.4 使用 HF_LEROBOT_HOME）
if [[ "$DATASET_REPO_ID" == ./* ]] || [[ "$DATASET_REPO_ID" == /* ]]; then
  ABS_PATH="$(cd "$DATASET_REPO_ID" 2>/dev/null && pwd || echo "$DATASET_REPO_ID")"
  export HF_LEROBOT_HOME="$(dirname "$ABS_PATH")"
  REPO_ID_NAME="$(basename "$ABS_PATH")"
  REPO_ARG="--dataset.repo_id=$REPO_ID_NAME"
else
  REPO_ARG="--dataset.repo_id=$DATASET_REPO_ID"
fi

# YeahBot 数据集只有 observation.images.top，映射为 policy 期望的 camera1；camera2/camera3 置空
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="$POLICY_REPO_ID" \
  $REPO_ARG \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.top": "observation.images.camera1"}' \
  --policy.empty_cameras=2 \
  --batch_size=$BATCH_SIZE \
  --steps=$STEPS \
  --output_dir=$OUTPUT_DIR \
  --job_name=$JOB_NAME \
  --policy.device=$POLICY_DEVICE \
  --wandb.enable=$WANDB_ENABLE \
  --policy.push_to_hub=false \
  $([[ -n "$OPTIMIZER_LR" ]] && echo "--optimizer.lr=$OPTIMIZER_LR")

echo "训练完成. 输出: $OUTPUT_DIR"
