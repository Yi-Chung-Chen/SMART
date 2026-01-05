#!/bin/bash
#SBATCH -A lusu                     # Your RCAC account
#SBATCH -p a100-80gb                # A100-80GB partition
#SBATCH --nodes=2                   # 2 nodes total
#SBATCH --ntasks-per-node=2         # 2 tasks per node (must match devices in config)
#SBATCH --gpus-per-node=2           # 2 A100 GPUs per node
#SBATCH --cpus-per-task=32          # CPUs for data loading workers
#SBATCH --mem=120G                  # Memory per node
#SBATCH --time=7-00:00:00            # 7 day time limit
#SBATCH --job-name=smart_n2_gpu2
#SBATCH --output=logs/smart_n2_gpu2_%j.out
#SBATCH --error=logs/smart_n2_gpu2_%j.err

# Create log and checkpoint directories
mkdir -p logs
mkdir -p checkpoints/n2_gpu2_run1

# Load RCAC environment
module load rcac
module list

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate conda environment (lowercase 'smart')
conda activate smart

# Print job info
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Print GPU info from all nodes
srun nvidia-smi --query-gpu=name,memory.total --format=csv

# NCCL environment variables for multi-node training
export NCCL_DEBUG=INFO              # Detailed logging (use WARN after first successful run)
export NCCL_IB_DISABLE=1            # Disable InfiniBand for testing (try 0 if this works)
export NCCL_ASYNC_ERROR_HANDLING=1  # Better error reporting
export NCCL_BLOCKING_WAIT=1         # More reliable for debugging
# export NCCL_SOCKET_IFNAME=ib0     # Commented out - let NCCL auto-detect
# export NCCL_IB_HCA=mlx5_0         # Commented out - let NCCL auto-detect

# PyTorch distributed settings
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Detailed distributed debugging

# Print debug info
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# Unbuffered Python output to see all rank outputs immediately
export PYTHONUNBUFFERED=1

# Add per-rank debug output
export SLURM_LABEL_OUTPUT=yes

# Run training with srun (handles multi-node coordination automatically)
srun python -u train.py \
  --config configs/train/train_a100_n2_gpu2.yaml \
  --save_ckpt_path checkpoints/n2_gpu2_run1

echo "Job finished at $(date)"
