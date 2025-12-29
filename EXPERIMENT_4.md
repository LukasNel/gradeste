# Experiment 4: Memory Optimization for Qwen3-4B

## Problem
CUDA OOM during Gumbel-Softmax `generate_soft` on H100 (80GB):
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 79.25 GiB of which 12.75 MiB is free.
Process 55 has 79.23 GiB memory in use. Of the allocated memory 77.84 GiB
is allocated by PyTorch, and 921.65 MiB is reserved by PyTorch but unallocated.
```

The `generate_soft` method builds a computation graph through `max_new_tokens` autoregressive steps, storing all intermediate activations for backprop. With 3 full Qwen3-4B models loaded (policy, reference, reward), memory exhausts.

## Changes

### 1. Reduced Batch Size (4 → 2)
```python
batch_size: int = 2  # was 4
gradient_accumulation_steps: int = 8  # was 4, increased to maintain effective batch
```
Halves per-step memory while maintaining effective batch size of 16.

### 2. Reduced Generation Length (48 → 32)
```python
max_new_tokens: int = 32  # was 48
```
Reduces sequence length by 33%, directly reducing activation memory in the autoregressive loop.

### 3. BFloat16 Precision (float32 → bfloat16)
```python
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model, 
    torch_dtype=torch.bfloat16,  # was float32
    attn_implementation="eager",  # avoid flash attention issues with soft tokens
)
```
Applied to all 3 models (policy, reference, reward). Halves memory for weights and activations.

### 4. Gradient Checkpointing
```python
base_model.gradient_checkpointing_enable()
```
Trades compute for memory by recomputing activations during backward pass instead of storing them.

### 5. Explicit Memory Management
```python
# Before generation
torch.cuda.empty_cache()

# After backward pass
del soft_tokens, soft_embeds, logits_seq, ref_logits, loss
torch.cuda.empty_cache()

# Inside generation loop
del outputs  # Free immediately after extracting logits
```
Prevents fragmentation and ensures tensors are freed promptly.

### 6. Autocast in Generation Loop
```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    outputs = self.model(...)
```
Ensures consistent precision through the autoregressive loop.

## Expected Memory Reduction

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Model weights (×3) | ~48 GB | ~24 GB | 50% |
| Activations | ~30 GB | ~10-15 GB | 50-66% |
| **Total** | ~78 GB | ~25-35 GB | ~55-65% |

## Trade-offs
- Slightly reduced generation diversity with shorter sequences
- ~10-20% slower training due to gradient checkpointing recomputation
- Potential minor numerical differences from bfloat16 (negligible in practice)

