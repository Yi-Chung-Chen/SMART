# Comparison: val.py vs eval.py vs eval_with_waymo_metrics.py

## Quick Summary

| Script | Purpose | When to Use | What It Does | Metrics |
|--------|---------|-------------|--------------|---------|
| **val.py** | Validation during training | During/after training | Monitor model on validation set | Token accuracy, loss |
| **eval.py** | Test evaluation | After training complete | Evaluate final model on test set | minADE, minFDE, saves predictions |
| **eval_with_waymo_metrics.py** | Full metric computation | After eval.py | Compute paper metrics from saved predictions | All Waymo Challenge metrics |

## Detailed Breakdown

### 1. val.py - Validation Script

**Purpose:** Check model performance during or right after training

**Key Characteristics:**
- Uses `trainer.validate()` (PyTorch Lightning validation mode)
- Runs on **validation set** (not test set)
- Uses **teacher forcing** by default (ground truth for next-token prediction)
- Designed for quick checks during training

**What It Computes:**
```python
# With inference_token = False (default):
- val_cls_acc: Token classification accuracy
- val_loss: Cross-entropy loss

# With inference_token = True:
- val_cls_acc: Token classification accuracy
- val_loss: Cross-entropy loss
- val_minADE: Average displacement error (basic implementation)
- val_minFDE: Final displacement error (basic implementation)
```

**Code Path in Model:**
```python
def validation_step(self, data, batch_idx):
    # Runs forward pass with teacher forcing
    pred = self(data)  # Not self.inference()

    # If inference_token=True, also runs:
    pred = self.inference(data)  # Autoregressive generation
```

**Typical Usage:**
```bash
# During training - automatic
python train.py --config configs/train/train_scalable.yaml

# After training - manual check
python val.py --config configs/validation/validation_scalable.yaml \
              --pretrain_ckpt epoch=31.ckpt
```

**Output Example:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      Validate metric      ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        val_cls_acc        │    0.5297578573226929     │
│         val_loss          │    3.4595344066619873     │
└───────────────────────────┴───────────────────────────┘
```

---

### 2. eval.py - Test Evaluation Script

**Purpose:** Evaluate trained model on test set and generate predictions

**Key Characteristics:**
- Uses `trainer.test()` (PyTorch Lightning test mode)
- Runs on **test set** (or validation set if test not available)
- **Always uses inference mode** (autoregressive generation)
- Generates predictions in Waymo submission format
- Can save predictions for later metric computation

**What It Computes:**
```python
- test_minADE: Average displacement error over trajectory
- test_minFDE: Final displacement error
- Saves predictions to disk (optional)
- Creates Waymo submission file (optional)
```

**Code Path in Model:**
```python
def test_step(self, data, batch_idx):
    # Always runs full inference
    pred = self.inference(data)

    # Creates JointScene (Waymo format)
    joint_scene = joint_scene_from_states(states, object_ids)

    # Stores for later evaluation
    self.test_predictions[scenario_id] = {...}

def on_test_epoch_end(self):
    # Computes metrics across all test scenarios
    # Saves predictions.pkl and submission.bin
```

**Typical Usage:**
```bash
# Basic evaluation (metrics only)
python eval.py \
  --config configs/evaluation/eval_test.yaml \
  --pretrain_ckpt epoch=31.ckpt

# Full evaluation (metrics + save predictions)
python eval.py \
  --config configs/evaluation/eval_test.yaml \
  --pretrain_ckpt epoch=31.ckpt \
  --save_predictions \
  --output_dir ./eval_results
```

**Output Example:**
```
================================================================================
Test Results Summary
================================================================================
minADE: 1.5234
minFDE: 2.8456
Total scenarios evaluated: 1500
================================================================================

Predictions saved to: ./eval_results/predictions.pkl
Waymo submission file saved to: ./eval_results/submission.bin

To evaluate with official Waymo metrics:
1. Use the Waymo Open Dataset evaluation toolkit
2. Or submit to: https://waymo.com/open/challenges/
```

**Files Created:**
- `eval_results/predictions.pkl` - Python pickle with all predictions
- `eval_results/submission.bin` - Waymo protobuf submission file

---

### 3. eval_with_waymo_metrics.py - Full Metrics Computation

**Purpose:** Compute official Waymo Challenge metrics from saved predictions

**Key Characteristics:**
- **Post-processing script** (doesn't run the model)
- Reads predictions from `eval.py` output
- Requires **original Waymo scenario files** (.tfrecord format)
- Uses Waymo's official metrics API
- Computes all 9 measurements across 3 categories

**What It Computes:**
```python
# Meta Metric (overall score)
- Realism Meta Metric: 0.7614

# Kinematic Metrics (4 measurements)
- Linear Speed
- Linear Acceleration
- Angular Speed
- Angular Acceleration

# Interactive Metrics (3 measurements)
- Distance to Nearest Object
- Collision Rate
- Time to Collision

# Map-based Metrics (2 measurements)
- Distance to Road Edge
- Off-road Rate

# Plus tie-breaker
- minADE
```

**Code Flow:**
```python
# 1. Load predictions from eval.py
predictions = pickle.load('eval_results/predictions.pkl')

# 2. For each scenario:
for scenario_id, pred in predictions.items():
    # Load original scenario from .tfrecord
    scenario = load_scenario_from_file(scenario_id)

    # Compute Waymo metrics
    metrics = waymo_metrics.compute_scenario_metrics_for_bundle(
        config, scenario, pred['joint_scene']
    )

# 3. Aggregate across all scenarios
final = waymo_metrics.aggregate_metrics(all_metrics)
```

**Typical Usage:**
```bash
# Must run eval.py first to generate predictions!

python eval_with_waymo_metrics.py \
  --predictions ./eval_results/predictions.pkl \
  --scenario_dir ./data/waymo/scenario/testing \
  --challenge 2024
```

**Output Example:**
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

---

## Workflow Comparison

### Typical Training Workflow (uses val.py)
```
1. python train.py --config ...
   ├─> Trains model
   └─> Automatically runs validation_step() each epoch
       └─> Reports: val_cls_acc, val_loss

2. python val.py --config ... --pretrain_ckpt epoch=31.ckpt
   └─> Manual validation check after training
       └─> Reports: val_cls_acc, val_loss
```

### Typical Evaluation Workflow (uses eval.py + eval_with_waymo_metrics.py)
```
1. python eval.py --config ... --pretrain_ckpt ... --save_predictions
   ├─> Loads model
   ├─> Runs inference on test set
   ├─> Computes: test_minADE, test_minFDE
   └─> Saves: predictions.pkl, submission.bin

2. python eval_with_waymo_metrics.py --predictions predictions.pkl ...
   ├─> Loads predictions from step 1
   ├─> Loads original scenarios
   ├─> Computes Waymo metrics using official API
   └─> Reports: Realism, Kinematic, Interactive, Map metrics
```

---

## Key Differences Summary

### Data Split
- **val.py**: Validation set
- **eval.py**: Test set
- **eval_with_waymo_metrics.py**: Uses eval.py's output

### Inference Mode
- **val.py**: Teacher forcing by default (`self(data)`)
- **eval.py**: Full autoregressive inference (`self.inference(data)`)
- **eval_with_waymo_metrics.py**: N/A (post-processing only)

### Metrics Computed
- **val.py**: Token accuracy, loss (+ basic minADE/minFDE if enabled)
- **eval.py**: minADE, minFDE
- **eval_with_waymo_metrics.py**: All 9 Waymo measurements + meta metric

### Model Execution
- **val.py**: ✅ Runs neural network
- **eval.py**: ✅ Runs neural network
- **eval_with_waymo_metrics.py**: ❌ No model execution (only metrics)

### Dependencies
- **val.py**: PyTorch, PyTorch Lightning
- **eval.py**: PyTorch, PyTorch Lightning, waymo-open-dataset (optional)
- **eval_with_waymo_metrics.py**: waymo-open-dataset (required), TensorFlow

### Speed
- **val.py**: Fast (teacher forcing)
- **eval.py**: Slow (autoregressive inference)
- **eval_with_waymo_metrics.py**: Medium (I/O bound, no model)

### Output Files
- **val.py**: None
- **eval.py**: predictions.pkl, submission.bin (optional)
- **eval_with_waymo_metrics.py**: Optional metrics file

---

## When to Use Which?

### Use `val.py` when:
- ✅ You want to quickly check model during/after training
- ✅ You need to verify training is working
- ✅ You want to compare checkpoints quickly
- ✅ You're debugging or iterating on model architecture

### Use `eval.py` when:
- ✅ Training is complete and you want final results
- ✅ You need to generate predictions for submission
- ✅ You want minADE/minFDE metrics
- ✅ You're preparing for Waymo server submission

### Use `eval_with_waymo_metrics.py` when:
- ✅ You've already run eval.py and saved predictions
- ✅ You need the full suite of Waymo Challenge metrics
- ✅ You want to reproduce paper results exactly
- ✅ You have original Waymo scenario files available

---

## Common Patterns

### Pattern 1: Quick Development Iteration
```bash
# Train
python train.py --config ...

# Quick check (uses validation_step automatically during training)
# Or manually:
python val.py --pretrain_ckpt epoch=31.ckpt
```

### Pattern 2: Final Evaluation for Paper
```bash
# Step 1: Generate predictions
python eval.py --pretrain_ckpt best_model.ckpt --save_predictions

# Step 2: Compute all metrics
python eval_with_waymo_metrics.py \
  --predictions ./eval_results/predictions.pkl \
  --scenario_dir ./data/waymo/scenario/testing
```

### Pattern 3: Server Submission
```bash
# Step 1: Generate submission file
python eval.py --pretrain_ckpt best_model.ckpt --save_predictions

# Step 2: Upload eval_results/submission.bin to Waymo website
# (No need for eval_with_waymo_metrics.py - server computes metrics)
```

---

## Architecture Diagram

```
Training Phase:
  train.py
    └─> validation_step (val.py logic)
        └─> val_cls_acc, val_loss

Evaluation Phase:
  eval.py
    ├─> test_step
    │   └─> Runs inference on test set
    │   └─> Generates predictions
    ├─> on_test_epoch_end
    │   └─> Computes minADE, minFDE
    │   └─> Saves predictions.pkl, submission.bin
    │
    └─> OUTPUT: predictions.pkl, submission.bin

Post-Processing:
  eval_with_waymo_metrics.py
    ├─> Loads predictions.pkl
    ├─> Loads scenario .tfrecord files
    ├─> Uses Waymo metrics API
    └─> OUTPUT: Full Waymo Challenge metrics
```

---

## Metric Comparison

| Metric | val.py | eval.py | eval_with_waymo_metrics.py |
|--------|--------|---------|----------------------------|
| Token Classification Accuracy | ✅ | ❌ | ❌ |
| Cross-Entropy Loss | ✅ | ❌ | ❌ |
| minADE | ⚠️ (optional) | ✅ | ✅ |
| minFDE | ⚠️ (optional) | ✅ | ❌ |
| Linear Speed | ❌ | ❌ | ✅ |
| Linear Acceleration | ❌ | ❌ | ✅ |
| Angular Speed | ❌ | ❌ | ✅ |
| Angular Acceleration | ❌ | ❌ | ✅ |
| Distance to Nearest | ❌ | ❌ | ✅ |
| Collision Rate | ❌ | ❌ | ✅ |
| Time to Collision | ❌ | ❌ | ✅ |
| Distance to Road Edge | ❌ | ❌ | ✅ |
| Off-road Rate | ❌ | ❌ | ✅ |
| Realism Meta Metric | ❌ | ❌ | ✅ |

---

## Bottom Line

- **val.py** = Quick validation check (training metrics)
- **eval.py** = Full test evaluation (basic trajectory metrics)
- **eval_with_waymo_metrics.py** = Official challenge metrics (paper numbers)

All three serve different purposes in the ML workflow!
