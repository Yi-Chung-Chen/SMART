#!/bin/bash
#SBATCH -A lusu                     # Your RCAC account
#SBATCH -p a100-80gb                # A100-80GB partition
#SBATCH --nodes=1                   # Single node
#SBATCH --ntasks-per-node=2         # 2 tasks (must match devices in config)
#SBATCH --gpus-per-node=2           # 2 A100 GPUs
#SBATCH --cpus-per-task=16          # CPUs per task (32 total / 2 tasks)
#SBATCH --mem=120G                  # Memory (reasonable for 2 GPUs)
#SBATCH --time=7-00:00:00            # 7 day limit
#SBATCH --job-name=smart_n1_gpu2
#SBATCH --output=logs/smart_n1_gpu2_%j.out
#SBATCH --error=logs/smart_n1_gpu2_%j.err

# Create log and checkpoint directories
mkdir -p logs
mkdir -p results

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

# Print GPU info
srun nvidia-smi --query-gpu=name,memory.total --format=csv

# PyTorch distributed settings (optional for single-node but doesn't hurt)
export MASTER_PORT=12355
export PYTHONUNBUFFERED=1

# Run training with srun for proper multi-GPU coordination
srun python -u train.py \
  --config configs/train/train_a100_n1_gpu2.yaml \
  --save_ckpt_path results

echo "Job finished at $(date)"
