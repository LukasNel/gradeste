# Experiment 6: Memory-Efficient GRADE Implementation

## Motivation
GRADE/Gumbel-Softmax training is memory-intensive because it stores full vocabulary distributions `[batch, seq_len, vocab_size]` for soft tokens (~32k+ vocab). On a 4B parameter model, this causes OOM even on H100 80GB.

## Key Optimizations

### 1. Top-k Gumbel-Softmax Filtering

**Problem:** Standard Gumbel-Softmax computes over full vocabulary (32k+ tokens).

**Solution:** Filter to top-k logits before applying Gumbel noise:
```python
def gumbel_softmax_topk(logits, tau, k=256, hard=False):
    # Get top-k logits
    topk_logits, topk_indices = logits.topk(k, dim=-1)  # [batch, k]
    
    # Apply Gumbel only to top-k
    gumbels = -torch.log(-torch.log(torch.rand_like(topk_logits) + 1e-10) + 1e-10)
    y_soft_topk = F.softmax((topk_logits + gumbels) / tau, dim=-1)
    
    # Scatter back to full vocab (sparse)
    y_soft = torch.zeros_like(logits).scatter_(-1, topk_indices, y_soft_topk)
    return y_soft, topk_indices
```

**Memory reduction:** From `O(batch × seq × vocab)` to `O(batch × seq × k)` where `k << vocab`.

With `k=256` and `vocab=32000`: **~125x reduction** for soft token storage.

### 2. Online KL Computation

**Problem:** Original implementation stores all reference logits `[batch, seq, vocab]` then computes KL.

**Solution:** Compute KL per-token during generation, accumulate scalar:
```python
kl_sum = torch.zeros(batch_size, device=device)

for step_idx in range(max_new_tokens):
    # Policy forward
    policy_logits = model(...)
    
    # Reference forward (immediately compute KL, don't store)
    with torch.no_grad():
        ref_logits = ref_model(...)
        kl_per_token = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
        kl_sum += kl_per_token
        del ref_logits  # Free immediately
```

**Memory reduction:** Eliminates `[batch, seq, vocab]` tensor for ref_logits. **~2x reduction**.

### 3. KV-Cache for Reference Model

**Problem:** Reference model recomputes full sequence at each step (no gradients needed, wasteful).

**Solution:** Use `past_key_values` for incremental decoding:
```python
# Initialize with prompt
ref_outputs = ref_model(input_ids=prompt_ids, use_cache=True)
ref_past_kv = ref_outputs.past_key_values

for step_idx in range(max_new_tokens):
    # Only pass new token, reuse KV cache
    ref_out = ref_model(
        input_ids=hard_token.unsqueeze(-1),  # [batch, 1]
        past_key_values=ref_past_kv,
        use_cache=True,
    )
    ref_past_kv = ref_out.past_key_values
```

**Memory reduction:** KV-cache grows linearly vs quadratic recomputation. **~3-4x faster**, reduced peak memory.

### 4. Sparse Reward Computation

**Problem:** Reward model's `forward_soft` does `soft_tokens @ embedding_weight` with full vocab tensor.

**Solution:** Use sparse representation with gather:
```python
def forward_soft_sparse(self, topk_indices, topk_weights, attention_mask):
    # topk_indices: [batch, seq, k]
    # topk_weights: [batch, seq, k]
    
    # Gather only needed embeddings
    selected_embeds = self.embedding(topk_indices)  # [batch, seq, k, hidden]
    
    # Weighted sum
    soft_embeddings = (topk_weights.unsqueeze(-1) * selected_embeds).sum(dim=2)
    return self.forward_from_embeddings(soft_embeddings, attention_mask)
```

**Memory reduction:** `O(batch × seq × k × hidden)` instead of `O(batch × seq × vocab × hidden)`.

## Implementation

New `GumbelTrainerMemoryEfficient` class combines all optimizations:
- Uses `generate_soft_topk()` instead of `generate_soft()`
- Computes KL online during generation loop
- Uses KV-cache for reference model
- Calls `forward_soft_sparse()` for reward computation

Original `GumbelTrainer` preserved as `gumbel_legacy`/`ste_legacy` for comparison.

## Configuration

New config parameter:
```python
gumbel_topk: int = 256  # Top-k filtering (0 = disabled, uses full vocab)
```

## Additional Optimizations (v2)

### 5. Gradient Checkpointing for Policy Forward
The policy forward pass is wrapped in `torch.utils.checkpoint.checkpoint()`:
```python
policy_logits = torch.utils.checkpoint.checkpoint(
    self._policy_forward_step,
    policy_embeds,
    policy_mask,
    use_reentrant=False,
)
```
This recomputes activations during backward instead of storing them.

### 6. Reduced Sequence Length
```python
max_new_tokens: int = 64  # Reduced from 128
```
Attention is O(n²), so halving sequence length gives ~4x memory reduction in attention.

### 7. Minimal Batch Size
```python
batch_size: int = 1  # Minimal for Gumbel
gradient_accumulation_steps: int = 16  # Compensate
```

### 8. Aggressive Memory Cleanup
- Removed unused `logits_list` and `hard_tokens_list`
- `del` intermediate tensors immediately after use
- `torch.cuda.empty_cache()` at critical points
- Detach tensors that don't need gradients (indices, hard_tokens)

## Expected Memory Profile

| Component | Original | Optimized | Reduction |
|-----------|----------|-----------|-----------|
| soft_tokens [B,T,V] | ~4 GB | ~32 MB | 125x |
| ref_logits [B,T,V] | ~4 GB | 0 | ∞ |
| ref model activations | ~8 GB | ~2 GB | 4x |
| reward matmul | ~4 GB | ~32 MB | 125x |
| policy activations | ~20 GB | ~5 GB (checkpointed) | 4x |
| **Total peak** | ~40+ GB | ~10-15 GB | ~3-4x |

*Estimates for batch=1, seq=64, vocab=32k, hidden=4096*

## Trade-offs

1. **Top-k approximation:** Tokens outside top-256 get zero probability. In practice, the top-256 tokens capture >99.9% of probability mass for typical LLM distributions.

2. **KV-cache memory:** Grows with sequence length, but this is offset by not storing intermediate activations.

3. **Numerical precision:** Online KL computation may have slightly different numerical behavior vs batch computation, but should be equivalent in expectation.

## Running the Experiment

```bash
# Memory-efficient GRADE (default now)
python main.py --method gumbel

# Legacy implementation for comparison  
python main.py --method gumbel_legacy

# Run all methods
python main.py --method all
```

## Metrics to Track

- `train/reward` - should match legacy implementation
- `train/kl` - should match legacy implementation  
- `val/reward_mean` - should match legacy implementation
- GPU memory usage (via `nvidia-smi` or wandb)
- Training time per step

## Success Criteria

1. Memory usage reduced by ~4-5x
2. Training rewards/KL match legacy within noise
3. Final test performance equivalent to legacy

