# Experiment 2: Bug Fixes and Configuration Changes

## Changes from Experiment 1

### 1. LoRA Made Optional (Full Fine-tuning by Default)

```python
# Config changes
use_lora: bool = False  # New flag, defaults to full fine-tuning
lora_r: int = 16
lora_alpha: int = 32
lora_dropout: float = 0.05
```

- `use_lora=False`: Full model fine-tuning (~355M params for gpt2-medium)
- `use_lora=True`: LoRA fine-tuning (~0.3M trainable params)

### 2. Reduced Training Steps

```python
max_steps: int = 1000  # Down from 2000
```

Rationale: 1000 steps should be sufficient to observe convergence trends. Can extend if methods are still improving.

### 3. PPO Trainer Fixes

**Critical bugs fixed:**

| Issue | Before | After |
|-------|--------|-------|
| Old log probs | Computed with gradients | Computed in `no_grad` block |
| KL calculation | Simple log ratio, could be negative | Proper per-sequence KL with masking |
| Log ratio clamp | ±20 (too loose) | ±2 (tighter, prevents explosion) |
| Value loss | Clipped value loss (unstable) | Simple MSE |
| Loss averaging | Mean over all elements | Proper masked averaging |
| Advantage norm | Global mean/std | Masked mean/std |

**New metrics logged:**
- `policy_loss`: Separate policy gradient loss
- `value_loss`: Separate value function loss

### 4. REINFORCE Trainer Fixes

**Bugs fixed:**

| Issue | Before | After |
|-------|--------|-------|
| Generation | No `no_grad` wrapper | Wrapped in `no_grad` |
| seq_log_probs | Sum (length-biased) | Mean (normalized by length) |
| KL gradient | Leaked through advantage | Fully in `no_grad` |
| KL scaling | Sum per sequence | Normalized by sequence length |
| Advantage | Raw (high variance) | Normalized `(adv - mean) / std` |

## Expected Improvements

1. **PPO stability**: Loss should stay in reasonable range (< 10), KL should be small positive (~0.01-0.1)
2. **REINFORCE variance**: Lower gradient variance due to advantage normalization
3. **Fair comparison**: All methods now have consistent KL penalty scaling

## Running the Experiment

```bash
python training_grade.py
```

Or with LoRA enabled:
```python
config = Config(use_lora=True, max_steps=1000)
```

## Metrics to Watch

- **reward**: Should increase for all methods
- **kl**: Should stay small and positive (0.01-0.5 range)
- **loss**: PPO < 10, REINFORCE may be higher but stable
- **grad_norm_mean**: Should be < 1.0 after clipping

