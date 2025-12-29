# Experiment 3: Larger Model with LoRA

## Changes from Experiment 2

### 1. Model Upgrade

```python
base_model: str = "Qwen/Qwen3-4B"  # Up from gpt2-medium
```

- ~10x larger model (4B vs 355M parameters)
- Better baseline capabilities and language understanding

### 2. LoRA Enabled

```python
use_lora: bool = True  # Changed from False
lora_r: int = 16
lora_alpha: int = 32
lora_dropout: float = 0.05
```

- Required for memory efficiency with larger model
- Trainable params: ~0.5% of full model

### 3. Extended Training

```python
max_steps: int = 2000  # Up from 1000
```

- Doubled training steps to give larger model more time to adapt

## Configuration Summary

| Parameter | Experiment 2 | Experiment 3 |
|-----------|--------------|--------------|
| Model | gpt2-medium | Qwen3-4B |
| Parameters | ~355M | ~4B |
| LoRA | Disabled | Enabled |
| Max Steps | 1000 | 2000 |

## Expected Outcomes

1. **Higher quality generations**: Larger model should produce more coherent text
2. **Better sentiment steering**: Improved understanding of task
3. **Longer training time**: ~4-10x slower per step due to model size

## Running the Experiment

```bash
python training_grade.py
```

## Notes

- Requires sufficient GPU VRAM (16GB+ recommended)
- LoRA keeps memory manageable despite larger model

