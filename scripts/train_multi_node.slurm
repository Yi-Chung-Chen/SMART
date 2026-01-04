#!/bin/bash
#SBATCH -A lusu                     # Your RCAC account
#SBATCH -p a100-80gb                # A100-80GB partition
#SBATCH --nodes=4                   # 4 nodes total
#SBATCH --ntasks-per-node=1         # 1 task (process) per node
#SBATCH --gpus-per-node=1           # 1 A100 GPU per node
#SBATCH --cpus-per-task=64          # CPUs for data loading workers
#SBATCH --time=48:00:00             # 48 hour time limit
#SBATCH --job-name=smart_n4_gpu1
#SBATCH --output=logs/smart_n4_gpu1_%j.out
#SBATCH --error=logs/smart_n4_gpu1_%j.err

# Create log directory
mkdir -p logs

# Load RCAC environment
module load rcac
module list

# Activate conda environment
source activate SMART

# Print job info
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# Print GPU info from all nodes
srun nvidia-smi --query-gpu=name,memory.total --format=csv

# NCCL environment variables for multi-node training
# Gilbreth has 100 Gbps InfiniBand - optimize for it!
export NCCL_DEBUG=INFO              # Detailed logging (use WARN after first successful run)
export NCCL_IB_DISABLE=0            # Enable InfiniBand
export NCCL_SOCKET_IFNAME=ib0       # InfiniBand interface
export NCCL_NET_GDR_LEVEL=5         # GPU Direct RDMA optimization
export NCCL_IB_HCA=mlx5_0           # InfiniBand adapter (may need adjustment)

# PyTorch distributed settings
export MASTER_PORT=12355

# Run training with srun (handles multi-node coordination automatically)
srun python train.py \
  --config configs/train/train_a100_n4_gpu1.yaml \
  --save_ckpt_path checkpoints/n4_gpu1_run1

echo "Job finished at $(date)"
