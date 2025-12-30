# Local Evaluation Guide for SMART

This guide explains how to evaluate your SMART model locally using Waymo Sim Agents Challenge metrics.

## Overview

The evaluation system provides two levels of metrics:

1. **Basic Metrics** (Always computed):
   - `minADE`: Minimum Average Displacement Error
   - `minFDE`: Minimum Final Displacement Error

2. **Waymo Challenge Metrics** (Optional, requires Waymo Open Dataset API):
   - Realism Meta Metric (overall score)
   - Kinematic Metrics (speed, acceleration)
   - Interactive Metrics (collisions, time-to-collision)
   - Map-based Metrics (off-road, distance to road edge)

## Installation

### Install Waymo Open Dataset API (Optional but Recommended)

```bash
conda activate smart
pip install waymo-open-dataset-tf-2-12-0==1.6.4
```

## Usage

### 1. Basic Evaluation (minADE/minFDE only)

```bash
python eval.py \
  --config configs/validation/validation_scalable.yaml \
  --pretrain_ckpt epoch=31.ckpt
```

This will output:
```
================================================================================
Test Results Summary
================================================================================
minADE: X.XXXX
minFDE: X.XXXX
Total scenarios evaluated: XXXX
================================================================================
```

### 2. Evaluation with Prediction Saving

To save predictions in Waymo submission format:

```bash
python eval.py \
  --config configs/validation/validation_scalable.yaml \
  --pretrain_ckpt epoch=31.ckpt \
  --output_dir ./eval_results \
  --save_predictions
```

This will create:
- `eval_results/predictions.pkl` - Predictions in pickle format
- `eval_results/submission.bin` - Waymo submission protobuf file

### 3. Create Evaluation Config

For full evaluation, create a test config `configs/evaluation/eval_test.yaml`:

```yaml
time_info: &time_info
  num_historical_steps: 11
  num_future_steps: 80
  token_size: 2048

Dataset:
  root:
  batch_size: 1
  shuffle: False  # Important: Don't shuffle test data
  num_workers: 8
  pin_memory: True
  persistent_workers: True
  val_raw_dir: ["data/waymo_processed/testing"]  # Use test set
  dataset: "scalable"
  <<: *time_info

Trainer:
  accelerator: "gpu"
  devices: 1

Model:
  mode: "validation"
  predictor: "smart"
  dataset: "waymo"
  input_dim: 2
  hidden_dim: 128
  output_dim: 2
  output_head: False
  num_heads: 8
  <<: *time_info
  head_dim: 16
  dropout: 0.1
  num_freq_bands: 64
  decoder:
    <<: *time_info
    num_map_layers: 3
    num_agent_layers: 6
    a2a_radius: 60
    pl2pl_radius: 10
    pl2a_radius: 30
    time_span: 30
```

Then run:

```bash
python eval.py \
  --config configs/evaluation/eval_test.yaml \
  --pretrain_ckpt your_checkpoint.ckpt \
  --save_predictions \
  --output_dir ./test_results
```

## Computing Waymo Challenge Metrics Locally

### Method 1: Using Preprocessed Scenarios (TODO)

Currently, the preprocessed `.pkl` files may not contain all scenario information needed for full Waymo metrics computation. This would require:

1. Access to original Waymo scenario files
2. Loading scenario data with map information
3. Using Waymo's `metric_features` and `metrics` APIs

Example code structure (to be implemented):

```python
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features, metrics

# Load metrics config
config = metrics.load_metrics_config('2024')

# For each scenario
for scenario_id, predictions in test_predictions.items():
    # Load original scenario data
    scenario = load_scenario(scenario_id)

    # Compute features
    features = metric_features.compute_metric_features(
        scenario, predictions['joint_scene']
    )

    # Compute metrics
    scenario_metrics = metrics.compute_scenario_metrics_for_bundle(
        config, scenario, [predictions['joint_scene']]
    )

# Aggregate across all scenarios
final_metrics = metrics.aggregate_metrics(all_scenario_metrics)
```

### Method 2: Using Waymo's Official Evaluation Toolkit

1. **Generate submission file:**
```bash
python eval.py --config configs/evaluation/eval_test.yaml \
               --pretrain_ckpt your_checkpoint.ckpt \
               --save_predictions
```

2. **Evaluate using Waymo toolkit:**
```bash
# Install the evaluation binary
# Follow instructions at: https://github.com/waymo-research/waymo-open-dataset

# Run evaluation
bazel-bin/waymo_open_dataset/wdl_limited/sim_agents_metrics/eval \
  --submission_file=./eval_results/submission.bin \
  --scenario_dir=./data/waymo/scenario/testing \
  --output_file=./eval_results/metrics.txt
```

### Method 3: Submit to Waymo Server

1. Generate submission: `eval_results/submission.bin`
2. Go to: https://waymo.com/open/challenges/
3. Upload your submission file
4. Wait for evaluation results (may take hours/days)

## Troubleshooting

### Issue: "Waymo metrics not available"

**Solution:** Install the Waymo Open Dataset API:
```bash
pip install waymo-open-dataset-tf-2-12-0==1.6.4
```

### Issue: "Could not compute Waymo metrics"

**Possible causes:**
1. Preprocessed data doesn't contain scenario information
2. Missing map data required for map-based metrics
3. Scenario files not available

**Solution:** Either:
- Use Method 2 or 3 above (official toolkit or server submission)
- Modify `data_preprocess.py` to preserve scenario data

### Issue: Out of memory during evaluation

**Solution:** Reduce batch size in config:
```yaml
Dataset:
  batch_size: 1  # Use 1 for evaluation
```

## Expected Results

Based on the paper, SMART models should achieve approximately:

| Model | Realism Meta Metric | minADE |
|-------|-------------------|--------|
| SMART-tiny | 0.7591 | ~1.5-2.0 |
| SMART-large | 0.7614 | ~1.3-1.8 |

Note: Exact values depend on:
- Model size and training
- Test set used
- Evaluation configuration

## Next Steps

After getting evaluation results:

1. **Compare with baselines** - Check if your metrics match expected values
2. **Analyze failure cases** - Look at scenarios with high errors
3. **Iterate on model** - Adjust training based on evaluation insights
4. **Submit to leaderboard** - Upload to Waymo for official ranking

## Additional Resources

- [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)
- [Waymo Sim Agents Challenge](https://waymo.com/open/challenges/)
- [SMART Paper](https://arxiv.org/abs/2405.15677)
- [Waymo Metrics Tutorial](https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_sim_agents.ipynb)
