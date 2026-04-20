#!/bin/bash
set -eo pipefail

# Navigate to your mounted workspace
cd /work

# Clone the repository and specific branch if it doesn't exist yet
if [ ! -d "RecurrentGemma" ]; then
    echo "Cloning repository..."
    git clone https://github.com/SW10-Cryptanalysis/RecurrentGemma.git
    cd RecurrentGemma
else
    echo "Git pulling newest changes..."
    cd RecurrentGemma
    git pull
fi

mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_CACHE_DIR="/work/.uv_cache"

# Dynamically count available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

# H100 NVLink & NCCL Optimizations
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [ "$NUM_GPUS" -gt 1 ]; then
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
fi

# Create venv if not already created
if [ ! -d ".venv" ]; then
    echo "Creating new uv virtual environment..."
    uv venv
fi

# Install project dependencies
uv pip install -e .

# Install hf_transfer to enable faster Hugging Face downloads
uv pip install hf_transfer

MASTER_PORT=$((10000 + $RANDOM % 20000))

echo "Launching torchrun with $NUM_GPUS processes..."
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m src.train "$@"

echo "Training Job finished at $(date)"
