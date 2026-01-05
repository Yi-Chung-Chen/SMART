#!/bin/bash
#SBATCH -A lusu                     # Your RCAC account
#SBATCH -p a100-80gb                # A100-80GB partition
#SBATCH --nodes=1                   # Single node
#SBATCH --ntasks=1                  # Single task
#SBATCH --gpus-per-node=4           # 4 A100 GPUs
#SBATCH --cpus-per-task=32          # CPUs for data loading
#SBATCH --mem=120G                  # Memory (reasonable for 4 GPUs)
#SBATCH --time=7-00:00:00            # 7 day limit
#SBATCH --job-name=smart_n1_gpu4
#SBATCH --output=logs/smart_n1_gpu4_%j.out
#SBATCH --error=logs/smart_n1_gpu4_%j.err

# Create log directory
mkdir -p logs

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
nvidia-smi

# Run training
python train.py \
  --config configs/train/train_a100_n1_gpu4.yaml \
  --save_ckpt_path checkpoints/n1_gpu4_run1

echo "Job finished at $(date)"
