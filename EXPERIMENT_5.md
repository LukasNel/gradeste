# Experiment 5: Longer Generations with Lower Learning Rate

## Changes from Experiment 4

| Parameter | Exp 4 | Exp 5 | Rationale |
|-----------|-------|-------|-----------|
| `max_new_tokens` | 32 | 128 | Longer generations for richer reward signal |
| `learning_rate` | 1e-4 | 1e-5 | More stable training with longer sequences |

## Hypothesis

- Longer sequences (128 tokens) provide more context for sentiment classification
- Lower learning rate (10x reduction) prevents instability from larger gradient variance with longer sequences
- Trade-off: slower convergence but potentially better final performance

## Memory Considerations

128 tokens is 4x the previous 32 tokens. The autoregressive generation loop stores activations for each step. With bfloat16 + gradient checkpointing from Exp 4, this should still fit in 80GB but will be tight. If OOM occurs, reduce batch_size to 1.

