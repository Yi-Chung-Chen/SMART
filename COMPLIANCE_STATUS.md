# Waymo Challenge Compliance Status

## Verification Results

✅ **Preprocessed test data**
❌ **NOT fully compliant with Waymo Challenge requirements**

After running `verify_test_compliance.py`, we found that:
- Test set `tracks_to_predict` (category 3) does NOT include all valid agents
- The model will only predict a subset of agents required by Waymo
- This means submissions would be **incomplete** for official challenge evaluation

## What This Means

### Current SMART Implementation
```python
# smart/modules/agent_decoder.py:495
agent_valid_mask[agent_category != 3] = False
```
- Only predicts agents in `tracks_to_predict` (category 3)
- Typically 4-8 agents per scenario
- Misses other valid agents (10-25 per scenario)

### Waymo Requirements
According to [official documentation](https://github.com/waymo-research/waymo-open-dataset/issues/799):
- Must simulate **ALL valid agents** at last history step
- Including AV/ego vehicle
- All agents participate in collision and offroad metrics

## Impact Analysis

### If You Keep Current Implementation

**Pros:**
- ✅ Matches the released SMART code exactly
- ✅ May match what was used for paper metrics
- ✅ Simpler to train (fewer agents to predict)
- ✅ Better accuracy on `tracks_to_predict` (focused training)

**Cons:**
- ❌ Submissions would be incomplete/invalid for Waymo Challenge
- ❌ Missing agents won't be evaluated in interactive metrics
- ❌ Can't get official leaderboard scores

### If You Modify to Predict All Agents

**Pros:**
- ✅ Valid Waymo Challenge submissions
- ✅ Can get official leaderboard scores
- ✅ All metrics (kinematic + interactive) computed correctly

**Cons:**
- ❌ Need to retrain model
- ❌ May have lower accuracy on non-`tracks_to_predict` agents
- ❌ Longer training time (more agents)
- ❌ May not match paper numbers exactly

## Possible Explanations for Discrepancy

### Why SMART won despite this issue:

1. **Separate submission pipeline**: Authors may have post-processed predictions to add missing agents
2. **Earlier Waymo version**: Challenge requirements may have changed between 2024 and code release
3. **Different codebase**: Released code might be simplified for public use
4. **Evaluation subset**: Paper metrics might only use `tracks_to_predict`, not full Waymo metrics

## Recommended Path Forward

### Phase 1: Training & Development (Current)
**Decision: Keep as-is** ✓

Use the current implementation:
- Train model on `category == 3` agents only
- Evaluate on validation set
- Get baseline minADE/minFDE metrics
- Reproduce paper results

### Phase 2: Decision Point (After Training)

Based on your goals, choose one of:

#### Option A: Paper Reproduction Only
**No changes needed**
- Continue with category 3 predictions
- Compare metrics to paper
- Use for development/research

#### Option B: Waymo Challenge Submission
**Modify the code**
- Implement full agent prediction
- Retrain or fine-tune model
- Generate compliant submissions
- Submit to Waymo leaderboard

#### Option C: Hybrid Approach
**Post-process predictions**
- Keep trained model as-is
- For missing agents: use simple baseline (constant velocity)
- Fill in gaps for submission
- Main model focuses on `tracks_to_predict`

## How to Modify When Ready

### Quick Fix (Inference Only)

If you want to predict all agents WITHOUT retraining:

**File: `smart/modules/agent_decoder.py`**
```python
# Line ~495, change from:
agent_valid_mask[agent_category != 3] = False

# To:
# Predict all agents valid at last history step
if self.training:
    # During training: only category 3 (as before)
    agent_valid_mask[agent_category != 3] = False
else:
    # During inference: all valid agents
    current_valid = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
    agent_valid_mask = current_valid[:, None].expand_as(agent_valid_mask)
```

**File: `smart/model/smart.py`**
```python
# Line ~397 in test_step, change from:
eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps-1]

# To:
# Predict all valid agents at last history step
eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps-1]
```

This allows you to:
- Train on category 3 only (faster, focused)
- Infer on all agents (compliant)
- Model may have lower accuracy on non-category-3 agents

### Full Fix (Training + Inference)

If you want to train on all agents:

**File: `smart/modules/agent_decoder.py`**
```python
# Line ~495, replace with:
# Always predict all valid agents
current_valid = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
agent_valid_mask = current_valid[:, None].expand_as(agent_valid_mask)
```

Then retrain from scratch.

## Current Status

```
┌─────────────────────────────────────────────────────┐
│ STATUS: Training with category 3 only               │
│                                                     │
│ ✓ Data preprocessed (train/val/test)              │
│ ✓ Evaluation scripts ready                        │
│ ✓ Compliance issue documented                     │
│ ⏳ Training not started yet                        │
│                                                     │
│ NEXT: Train baseline model, then decide           │
└─────────────────────────────────────────────────────┘
```

## Documentation

- Compliance check: `check_waymo_compliance.py`
- Test verification: `verify_test_compliance.py`
- This status doc: `COMPLIANCE_STATUS.md`

## References

- [Waymo Challenge Requirements](https://waymo.com/open/challenges/)
- [GitHub Issue #799](https://github.com/waymo-research/waymo-open-dataset/issues/799) - Official clarification
- [SMART Paper](https://arxiv.org/abs/2405.15677)

---

**Last Updated:** Based on test data verification
**Decision:** Proceeding with training on category 3 only
**Review:** Re-evaluate after initial training results
