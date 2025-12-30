# Quick Start: Evaluating SMART

## TL;DR - Run Evaluation Now

### Basic Evaluation (Quick)

```bash
# Evaluate with minADE/minFDE only
python eval.py \
  --config configs/validation/validation_scalable.yaml \
  --pretrain_ckpt epoch=31.ckpt
```

### Full Evaluation (Recommended)

```bash
# Step 1: Generate predictions
python eval.py \
  --config configs/evaluation/eval_test.yaml \
  --pretrain_ckpt epoch=31.ckpt \
  --save_predictions \
  --output_dir ./eval_results

# Step 2: Compute Waymo metrics (requires original scenario files)
python eval_with_waymo_metrics.py \
  --predictions ./eval_results/predictions.pkl \
  --scenario_dir ./data/waymo/scenario/testing \
  --challenge 2024
```

## What You Get

### Option 1: Basic Metrics (Always Available)

Running `eval.py` gives you:

```
================================================================================
Test Results Summary
================================================================================
minADE: 1.5234
minFDE: 2.8456
Total scenarios evaluated: 11
================================================================================
```

### Option 2: Full Waymo Challenge Metrics

Running `eval_with_waymo_metrics.py` gives you:

```
================================================================================
WAYMO SIM AGENTS CHALLENGE EVALUATION RESULTS
================================================================================

REALISM META METRIC: 0.7614
  (Primary ranking metric - higher is better)

KINEMATIC METRICS: 0.8123
  Linear Speed:         0.8456
  Linear Acceleration:  0.8234
  Angular Speed:        0.7891
  Angular Acceleration: 0.7912

INTERACTIVE METRICS: 0.7456
  Distance to Nearest:  0.8123
  Collision Rate:       0.7234
  Time to Collision:    0.7012

MAP-BASED METRICS: 0.7289
  Distance to Road Edge: 0.7456
  Off-road Rate:         0.7123

minADE: 1.5234
  (Tie-breaker metric - lower is better)
================================================================================
```

## Three Evaluation Methods

### Method 1: Quick Test (No Waymo API)

**Use when:** You just want basic metrics quickly

```bash
python eval.py --config configs/validation/validation_scalable.yaml \
               --pretrain_ckpt epoch=31.ckpt
```

**Pros:** Fast, no dependencies
**Cons:** Only minADE/minFDE, not paper metrics

### Method 2: Local Full Evaluation (Requires Waymo API)

**Use when:** You have original scenario files and want all metrics

```bash
# Install Waymo API first
pip install waymo-open-dataset-tf-2-12-0==1.6.4

# Generate predictions
python eval.py --config configs/evaluation/eval_test.yaml \
               --pretrain_ckpt epoch=31.ckpt \
               --save_predictions

# Compute all metrics
python eval_with_waymo_metrics.py \
  --predictions ./eval_results/predictions.pkl \
  --scenario_dir ./data/waymo/scenario/testing
```

**Pros:** All metrics, unlimited runs, immediate results
**Cons:** Requires original scenario files

### Method 3: Server Submission (Official)

**Use when:** You want official leaderboard scores

```bash
# Generate submission file
python eval.py --config configs/evaluation/eval_test.yaml \
               --pretrain_ckpt epoch=31.ckpt \
               --save_predictions

# Upload eval_results/submission.bin to:
# https://waymo.com/open/challenges/
```

**Pros:** Official ranking, no local setup
**Cons:** Slow, limited submissions

## File Structure After Evaluation

```
SMART/
├── eval.py                      # Basic evaluation script (created)
├── eval_with_waymo_metrics.py   # Full metrics script (created)
├── configs/evaluation/
│   └── eval_test.yaml          # Evaluation config (created)
├── eval_results/               # Created when you run with --save_predictions
│   ├── predictions.pkl         # Predictions for post-processing
│   └── submission.bin          # Waymo submission file
└── EVALUATION.md               # Detailed documentation (created)
```

## Troubleshooting

**Q: "Waymo metrics not available"**
A: Run: `pip install waymo-open-dataset-tf-2-12-0==1.6.4`

**Q: "Scenario not found"**
A: You need original .tfrecord files in `--scenario_dir`, not just .pkl files

**Q: "Out of memory"**
A: Set `batch_size: 1` in your config

**Q: "How do I know if my model is good?"**
A: Compare to paper baselines:
- SMART-tiny: Realism ~0.76, minADE ~1.5-2.0
- SMART-large: Realism ~0.76, minADE ~1.3-1.8

## Next Steps

1. **Run basic eval** to verify everything works
2. **Run full eval** to get all metrics
3. **Compare with paper** to check reproduction
4. **Submit to server** for official ranking (optional)

For more details, see `EVALUATION.md`.
