# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMART (Scalable Multi-agent Real-time Motion Generation via Next-token Prediction) is an autonomous driving motion generation system that models vectorized map and agent trajectory data into discrete sequence tokens. The project won the Waymo Open Sim Agents Challenge 2024 and was accepted to NeurIPS 2024.

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate SMART

# Install additional dependencies
pip install -r requirements.txt

# If PyG installation fails
bash scripts/install_pyg.sh
```

**Key Dependencies:**
- Python 3.9
- PyTorch 1.12.1 with CUDA 11.3
- PyTorch Lightning 2.0.3
- PyTorch Geometric 2.6.1
- Waymo Open Dataset API (waymo-open-dataset-tf-2-12-0==1.6.4)

## Data Setup

The project uses Waymo Open Motion Dataset in scenario protocol format:

```bash
# Download and organize data into:
# data/waymo/scenario/{training,validation,testing}

# Preprocess the dataset
python data_preprocess.py \
  --input_dir ./data/waymo/scenario/training \
  --output_dir ./data/waymo_processed/training
```

Processed data goes to `data/waymo_processed/{training,validation,testing}`.

## Common Commands

### Training
```bash
# Train with default config
python train.py --config configs/train/train_scalable.yaml

# Train with pretrained checkpoint
python train.py --config ${config_path} --pretrain_ckpt ${ckpt_path}

# Resume from checkpoint
python train.py --config ${config_path} --ckpt_path ${ckpt_path}

# Specify checkpoint save path
python train.py --config ${config_path} --save_ckpt_path ${path}
```

### Evaluation
```bash
# Validate model
python val.py --config configs/validation/validation_scalable.yaml --pretrain_ckpt ${ckpt_path}
```

Note: There's also an `eval.py` script mentioned in README but not present in the codebase.

## Architecture

### Core Model Components

**SMART (smart/model/smart.py):**
- Main PyTorch Lightning module
- Loads trajectory tokens from `smart/tokens/cluster_frame_5_2048.pkl`
- Loads map tokens from `smart/tokens/map_traj_token5.pkl`
- Contains encoder (SMARTDecoder), metrics (minADE, minFDE, TokenCls)
- Handles training/validation/test steps with next-token prediction

**SMARTDecoder (smart/modules/smart_decoder.py):**
- Top-level decoder combining map and agent processing
- Two sub-modules:
  - `SMARTMapDecoder`: Processes map polylines with attention layers
  - `SMARTAgentDecoder`: Processes agent trajectories with next-token prediction
- Provides `forward()` for training and `inference()` for evaluation

**Data Pipeline:**
- `MultiDataModule` (smart/datamodules/): PyTorch Lightning DataModule wrapper
- `MultiDataset` (smart/datasets/scalable_dataset.py): Handles loading preprocessed .pkl files
- `WaymoTargetBuilder` (smart/transforms/): Transforms raw data for training

### Key Architectural Patterns

**Next-Token Prediction:** The model discretizes continuous trajectories into tokens (token_size=2048 by default) and predicts them sequentially, similar to language models.

**Graph-Based Encoding:** Uses PyTorch Geometric with heterogeneous graph data (HeteroData) to represent:
- Map polylines (pl) and their relationships
- Agents (vehicles, pedestrians, cyclists)
- Spatial relationships via radius-based edges (a2a_radius=60, pl2a_radius=30, pl2pl_radius=10)

**Multi-Layer Attention:**
- Map layers (default: 3) for polyline encoding
- Agent layers (default: 6) for trajectory encoding
- Uses custom attention layers (smart/layers/attention_layer.py) with Fourier embeddings

### Config System

Configs are YAML files in `configs/{train,validation}/` with this structure:
- `Dataset`: Data paths, batch sizes, num_workers
- `Trainer`: PyTorch Lightning trainer settings (devices, epochs, strategy)
- `Model`: Architecture hyperparameters (hidden_dim, num_layers, radii, etc.)

Time settings use YAML anchors (`&time_info`, `<<: *time_info`) for consistency across sections.

## Important Implementation Details

**Coordinate Systems:** The codebase works with global coordinates for Waymo data. The `data_preprocess.py` script transforms from Waymo's scenario format to the model's expected format.

**Token-Based Representation:**
- Trajectory tokens are pre-clustered and stored in `smart/tokens/cluster_frame_5_2048.pkl`
- Map tokens stored in `smart/tokens/map_traj_token5.pkl`
- These must exist before training

**DDP Training:** Training uses DistributedDataParallel with `find_unused_parameters=True` and `gradient_as_bucket_view=True` for multi-GPU support.

**Checkpoint Format:**
- Model checkpoints monitored on `val_cls_acc` (classification accuracy)
- Saves top 5 checkpoints by default
- Load via `model.load_params_from_file(filename=ckpt_path, logger=logger)`

**Metrics:**
- minADE: Minimum Average Displacement Error
- minFDE: Minimum Final Displacement Error
- TokenCls: Next-token classification accuracy

## File Organization

- `smart/model/`: Main model definitions
- `smart/modules/`: Decoder modules (map, agent, combined)
- `smart/layers/`: Neural network layers (attention, MLP, Fourier embeddings)
- `smart/datasets/`: Dataset loading and preprocessing
- `smart/datamodules/`: PyTorch Lightning data modules
- `smart/transforms/`: Data transformation utilities
- `smart/metrics/`: Evaluation metrics
- `smart/utils/`: Utilities (config loading, logging, geometry, graph operations)
- `smart/tokens/`: Pre-computed trajectory and map tokens
- `configs/`: Training and validation configuration files
- `scripts/`: Helper scripts (PyG installation, trajectory clustering)
