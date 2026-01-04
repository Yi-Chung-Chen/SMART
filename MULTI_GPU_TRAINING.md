# Multi-GPU Training Guide for SMART on RCAC Gilbreth

This guide provides complete instructions for training the SMART model using multiple GPUs on Purdue's RCAC Gilbreth cluster.

## Overview

Gilbreth provides excellent hardware for distributed training:
- **GPUs**: A100-80GB across multiple sub-clusters (G, I, J, K, N)
- **Best Configuration**: Sub-cluster J has nodes with 4× A100-80GB + NVLink
- **Network**: 100 Gbps InfiniBand for multi-node communication
- **Your Account**: `lusu`
- **GPU Partition**: `a100-80gb`

## Available Configurations

This repository includes two pre-configured setups for 4-GPU training:

| Configuration | Config File | SLURM Script | Expected Scaling | Best For |
|---------------|-------------|--------------|------------------|----------|
| **1 node × 4 GPUs** | `train_a100_n1_gpu4.yaml` | `train_single_node_4gpu.slurm` | 95-99% (NVLink) | Best performance |
| **4 nodes × 1 GPU** | `train_a100_n4_gpu1.yaml` | `train_multi_node.slurm` | 85-95% (InfiniBand) | Easier allocation |

Both configurations use:
- Effective batch size: 64 (16 per GPU × 4 GPUs)
- Workers: 64 per dataloader
- Max epochs: 32
- Learning rate: 0.0005

## Quick Start

### Option 1: Single-Node 4-GPU (Recommended - Best Performance)

```bash
# Submit job
sbatch scripts/train_single_node_4gpu.slurm

# Check queue
squeue -u $USER

# Monitor training (replace JOBID)
tail -f logs/smart_n1_gpu4_JOBID.out
```

**Pros**: Best performance with NVLink inter-GPU communication (~99% scaling efficiency)

**Cons**: May be harder to allocate (limited to nodes with 4 GPUs)

### Option 2: Multi-Node 4×1 GPU (Fallback)

```bash
# Submit job
sbatch scripts/train_multi_node.slurm

# Check queue
squeue -u $USER

# Monitor training (replace JOBID)
tail -f logs/smart_n4_gpu1_JOBID.out
```

**Pros**: Easier to allocate (single-GPU nodes more available), still excellent performance with 100 Gbps InfiniBand

**Cons**: Slightly lower scaling efficiency (~90%) due to inter-node communication

### Option 3: Hybrid Approach (Optimal Strategy)

Submit both jobs and use whichever allocates first:

```bash
# Submit both jobs
JOB1=$(sbatch scripts/train_single_node_4gpu.slurm | awk '{print $4}')
JOB2=$(sbatch scripts/train_multi_node.slurm | awk '{print $4}')

echo "Single-node job: $JOB1"
echo "Multi-node job: $JOB2"

# Check which starts first
squeue -u $USER

# Cancel the one that didn't start (if the other is already running)
# scancel $JOB1  # or scancel $JOB2
```

## Detailed Configuration

### Single-Node 4-GPU Setup

**Config**: `configs/train/train_a100_n1_gpu4.yaml`
```yaml
Trainer:
  devices: 4        # 4 GPUs on single node
  num_nodes: 1      # Single node
```

**SLURM Script**: `scripts/train_single_node_4gpu.slurm`
```bash
#SBATCH -A lusu                     # Account
#SBATCH -p a100-80gb                # Partition
#SBATCH --nodes=1                   # 1 node
#SBATCH --gpus-per-node=4           # 4 GPUs
#SBATCH --cpus-per-task=64          # 64 CPUs
#SBATCH --time=48:00:00             # 48 hour limit
```

### Multi-Node 4×1 GPU Setup

**Config**: `configs/train/train_a100_n4_gpu1.yaml`
```yaml
Trainer:
  devices: 1        # 1 GPU per node
  num_nodes: 4      # 4 nodes total
```

**SLURM Script**: `scripts/train_multi_node.slurm`
```bash
#SBATCH -A lusu                     # Account
#SBATCH -p a100-80gb                # Partition
#SBATCH --nodes=4                   # 4 nodes
#SBATCH --gpus-per-node=1           # 1 GPU per node
#SBATCH --cpus-per-task=64          # 64 CPUs per task
#SBATCH --time=48:00:00             # 48 hour limit
```

The multi-node script also configures NCCL for optimal InfiniBand performance:
```bash
export NCCL_DEBUG=INFO              # Detailed logging
export NCCL_IB_DISABLE=0            # Enable InfiniBand
export NCCL_SOCKET_IFNAME=ib0       # InfiniBand interface
export NCCL_NET_GDR_LEVEL=5         # GPU Direct RDMA
export NCCL_IB_HCA=mlx5_0           # InfiniBand adapter
```

## Monitoring and Management

### Check Job Status
```bash
# View your jobs in queue
squeue -u $USER

# Detailed job info
scontrol show job JOBID

# View all A100 GPU jobs
squeue -p a100-80gb
```

### Monitor Training Progress
```bash
# Follow log output in real-time
tail -f logs/smart_n1_gpu4_JOBID.out

# Check GPU utilization (once job starts, SSH to compute node)
ssh NODE_NAME  # Get node name from squeue
nvidia-smi dmon
```

### Cancel Jobs
```bash
# Cancel specific job
scancel JOBID

# Cancel all your jobs
scancel -u $USER
```

## Expected Performance

### Single-Node 4-GPU
- **Scaling Efficiency**: 95-99%
- **Training Speed**: ~4× faster than single GPU
- **Communication**: NVLink inter-GPU (~600 GB/s)
- **Best for**: Maximum training throughput

### Multi-Node 4×1 GPU
- **Scaling Efficiency**: 85-95%
- **Training Speed**: ~3.5× faster than single GPU
- **Communication**: 100 Gbps InfiniBand
- **Best for**: Easier allocation, still excellent performance

### Effective Batch Size
Both configurations achieve the same effective batch size:
- 16 (per GPU) × 4 (GPUs) = **64 total batch size**

## Troubleshooting

### Job Doesn't Start (Pending in Queue)

**Check queue position:**
```bash
squeue -u $USER
```

**Common reasons:**
- Resource constraints: Try the multi-node config (easier allocation)
- Account limits: Check with `sacctmgr show assoc user=$USER`

**Solution**: If single-node job is pending too long, cancel and use multi-node:
```bash
scancel JOBID
sbatch scripts/train_multi_node.slurm
```

### NCCL Errors in Multi-Node Training

**Symptom**: Errors like "NCCL initialization failed" or "Network unreachable"

**Common causes:**
- Incorrect InfiniBand interface name
- Wrong NCCL settings for the cluster

**Solutions:**
1. Check NCCL debug output in logs for actual interface names
2. Adjust `NCCL_SOCKET_IFNAME` in the SLURM script (try `ib0`, `eth0`, etc.)
3. Adjust `NCCL_IB_HCA` based on actual adapter name
4. Contact RCAC support for cluster-specific settings

### Low GPU Utilization (<80%)

**Possible causes:**
- Data loading bottleneck
- Too much communication overhead (multi-node)

**Solutions:**
1. Check if CPUs are saturating: `top` on compute node
2. Increase `num_workers` in config if CPUs available
3. Increase batch size (A100-80GB has plenty of memory)
4. Use gradient accumulation to reduce communication frequency

### Out of Memory (OOM) Error

**Solutions:**
1. Reduce batch size in config (try 8 or 12 instead of 16)
2. Enable gradient accumulation:
   ```yaml
   Trainer:
     accumulate_grad_batches: 2  # Effective batch = 8 GPUs worth
   ```

### Training Slower Than Expected

**Check:**
1. GPU utilization: Should be >90%
2. NCCL using InfiniBand (check logs for "NET/IB" messages)
3. Data loading not bottlenecking (CPU usage)

**Benchmark single GPU first:**
```bash
sbatch scripts/train_a100_n1_gpu1.yaml  # Use existing single-GPU config
```
Then compare multi-GPU speedup.

## Resuming from Checkpoint

If your job times out or you need to resume training:

```bash
# Modify the command in SLURM script to include --ckpt_path
python train.py \
  --config configs/train/train_a100_n1_gpu4.yaml \
  --ckpt_path checkpoints/n1_gpu4_run1/last.ckpt \
  --save_ckpt_path checkpoints/n1_gpu4_run1
```

Or create a new SLURM script with the checkpoint path added.

## Customization

### Adjust Batch Size

Edit the config file (e.g., `configs/train/train_a100_n1_gpu4.yaml`):
```yaml
Dataset:
  train_batch_size: 24  # Increase from 16 (effective batch = 96)
```

### Adjust Training Duration

```yaml
Trainer:
  max_epochs: 64  # Increase from 32
```

### Change Time Limit

Edit SLURM script:
```bash
#SBATCH --time=96:00:00  # Increase to 96 hours
```

### Gradient Accumulation

To increase effective batch size without using more memory:
```yaml
Trainer:
  accumulate_grad_batches: 2  # Effective batch = 16 × 4 × 2 = 128
```

## Performance Optimization Tips

### 1. Maximize GPU Utilization
- Increase batch size until GPU memory is 80-90% full
- Use mixed precision training (already set to `precision: 32`, can try `16`)

### 2. Optimize Data Loading
- Ensure `num_workers` matches available CPU cores
- Use `pin_memory: True` (already enabled)
- Use `persistent_workers: True` (already enabled)

### 3. Reduce Communication Overhead (Multi-Node)
- Use gradient accumulation to reduce all-reduce frequency
- Increase batch size to compute/communication ratio
- Verify InfiniBand is being used (check NCCL logs)

### 4. Monitor and Benchmark
- Always run single-GPU baseline first
- Calculate scaling efficiency: (multi-GPU throughput) / (single-GPU throughput × num_GPUs)
- Target: >85% for multi-node, >95% for single-node

## Other Existing Configurations

The repository also includes these configurations:

| Config | Nodes | GPUs/Node | Total GPUs | Use Case |
|--------|-------|-----------|------------|----------|
| `train_a100_n1_gpu1.yaml` | 1 | 1 | 1 | Baseline/testing |
| `train_a100_n1_gpu2.yaml` | 1 | 2 | 2 | Small-scale training |
| `train_a100_n2_gpu1.yaml` | 2 | 1 | 2 | Multi-node testing |
| `train_a100_n2_gpu2.yaml` | 2 | 2 | 4 | Balanced approach |

## Additional Resources

### RCAC Documentation
- Gilbreth Overview: https://www.rcac.purdue.edu/compute/gilbreth
- Running Jobs: https://www.rcac.purdue.edu/knowledge/gilbreth/run
- GPU Usage: https://www.rcac.purdue.edu/knowledge/gilbreth/run/examples/slurm/gpu

### PyTorch Lightning
- Multi-GPU Training: https://lightning.ai/docs/pytorch/stable/accelerators/gpu.html
- Multi-Node Training: https://lightning.ai/docs/pytorch/stable/clouds/cluster.html

### NCCL
- Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

## Support

For issues with:
- **SLURM/Gilbreth**: Contact RCAC support at rcac-help@purdue.edu
- **SMART model**: Check repository issues or documentation
- **Multi-GPU setup**: Refer to troubleshooting section above

## Summary

**Quick Decision Guide:**

1. **First time?** → Try single-node 4-GPU (`train_single_node_4gpu.slurm`)
2. **Can't get single-node?** → Use multi-node 4×1 (`train_multi_node.slurm`)
3. **Not sure?** → Submit both, use whichever starts first
4. **Need maximum throughput?** → Single-node 4-GPU (best scaling)
5. **Need quick allocation?** → Multi-node 4×1 (easier to get)

Both configurations will give you excellent performance - the choice is primarily about allocation time vs. slightly better scaling efficiency.
