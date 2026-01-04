#!/bin/bash
#SBATCH -A lusu                     # Your RCAC account
#SBATCH -p a100-80gb                # A100-80GB partition
#SBATCH --nodes=1                   # Single node
#SBATCH --ntasks=1                  # Single task
#SBATCH --gpus-per-node=4           # 4 A100 GPUs
#SBATCH --cpus-per-task=64          # CPUs for data loading
#SBATCH --time=48:00:00             # 48 hour limit
#SBATCH --job-name=smart_n1_gpu4
#SBATCH --output=logs/smart_n1_gpu4_%j.out
#SBATCH --error=logs/smart_n1_gpu4_%j.err

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
nvidia-smi

# Run training
python train.py \
  --config configs/train/train_a100_n1_gpu4.yaml \
  --save_ckpt_path checkpoints/n1_gpu4_run1

echo "Job finished at $(date)"
