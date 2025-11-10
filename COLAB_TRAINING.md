# Google Colab Training Guide

## Setup (Run in first cell)

```bash
# Clone repository
!git clone https://github.com/madst0614/plastic-neural-networks.git
%cd plastic-neural-networks

# Checkout experiment branch
!git checkout claude/refiner-experiments-ffn-011CUz2zZ6v1XqK5Fzwdy8eN

# Install package and dependencies
!pip install -e .
```

---

## Experiment 1: Dual Attention + Dual FFN (61M params)

**Structure**: Each refiner has 2 transformer blocks (Attention + FFN)

```bash
!python scripts/train.py \
  --model pnn_exp1 \
  --max_samples 200000 \
  --batch_size 256 \
  --gradient_accumulation 4 \
  --epochs 15 \
  --lr 3e-4 \
  --warmup_steps 500 \
  --checkpoint_dir checkpoints/exp1_dual_blocks \
  --use_amp \
  --use_tf32
```

**Expected size**: 55M → 61M (+5.55M)

---

## Experiment 2: Dual Refiners (68M params)

**Structure**: 2 independent refiners applied alternately

```bash
!python scripts/train.py \
  --model pnn_exp2 \
  --max_samples 200000 \
  --batch_size 256 \
  --gradient_accumulation 4 \
  --epochs 15 \
  --lr 3e-4 \
  --warmup_steps 500 \
  --checkpoint_dir checkpoints/exp2_dual_refiners \
  --use_amp \
  --use_tf32
```

**Expected size**: 55M → 68M (+13.4M)

---

## Baseline PNN (55M params)

**For comparison**

```bash
!python scripts/train.py \
  --model pnn \
  --max_samples 200000 \
  --batch_size 256 \
  --gradient_accumulation 4 \
  --epochs 15 \
  --lr 3e-4 \
  --warmup_steps 500 \
  --checkpoint_dir checkpoints/baseline_pnn \
  --use_amp \
  --use_tf32
```

---

## Monitor Training (Optional)

```python
# Load tensorboard (run in separate cell)
%load_ext tensorboard
%tensorboard --logdir checkpoints/
```

---

## Quick Test (Verify models work)

```python
import torch
from pnn.models.pnn import create_pnn_model

# Test all models
for model_type in ['pnn', 'pnn_exp1', 'pnn_exp2']:
    print(f"\n{'='*60}")
    model = create_pnn_model(model_type=model_type)

    # Quick forward pass
    input_ids = torch.randint(0, 30522, (2, 128))
    attention_mask = torch.ones(2, 128)
    output = model(input_ids, attention_mask)

    print(f"✓ {model_type.upper()} output shape: {output.shape}")
```

---

## Training Tips

### Batch Size Adjustment
If you get OOM (Out of Memory):
```bash
# Reduce batch size, increase gradient accumulation
--batch_size 128 \
--gradient_accumulation 8
```

### For Free Colab (T4 GPU)
```bash
# Conservative settings
--batch_size 128 \
--gradient_accumulation 8 \
--max_samples 100000
```

### For Colab Pro (A100 GPU)
```bash
# Aggressive settings
--batch_size 512 \
--gradient_accumulation 2 \
--max_samples 500000
```

---

## Expected Training Time

**On T4 GPU (Free Colab):**
- ~6-8 hours for 200K samples, 15 epochs

**On A100 GPU (Colab Pro):**
- ~2-3 hours for 200K samples, 15 epochs

---

## Resume Training

If training is interrupted:

```bash
!python scripts/train.py \
  --model pnn_exp1 \
  --resume checkpoints/exp1_dual_blocks/checkpoint_epochX.pt \
  --epochs 15 \
  ...
```

---

## Compare Results

```python
import json

# Load metrics
with open('checkpoints/baseline_pnn/metrics.json') as f:
    baseline = json.load(f)

with open('checkpoints/exp1_dual_blocks/metrics.json') as f:
    exp1 = json.load(f)

with open('checkpoints/exp2_dual_refiners/metrics.json') as f:
    exp2 = json.load(f)

print(f"Baseline:  {baseline['best_accuracy']:.4f}")
print(f"Exp1:      {exp1['best_accuracy']:.4f}")
print(f"Exp2:      {exp2['best_accuracy']:.4f}")
```
