# EXPERIMENT 1: GRADE vs Policy Gradient Methods for LLM Alignment

## Summary

**Objective**: Compare differentiable relaxation methods (GRADE) against policy gradient methods (PPO, REINFORCE) for steering LLM generations toward positive sentiment.

**Task**: Sentiment-controlled text generation on IMDB movie reviews.

**Key Hypothesis**: Gumbel-Softmax relaxation enables direct backpropagation through the generation process, potentially offering lower gradient variance and better sample efficiency compared to policy gradient estimators.

| Method | Type | Gradient Estimator |
|--------|------|-------------------|
| GRADE | Differentiable | Gumbel-Softmax (soft) |
| GRADE-STE | Differentiable | Straight-Through Estimator |
| PPO | Policy Gradient | Clipped surrogate objective |
| REINFORCE | Policy Gradient | Vanilla PG with baseline |

---

## 1. Methods

### 1.1 GRADE (Gumbel-Softmax)

Replaces discrete token sampling with a differentiable relaxation:

```
y_soft = softmax((logits + gumbel_noise) / τ)
```

- **Temperature annealing**: τ starts at 2.0, linearly decays to 0.5 over 1000 steps
- **Soft embeddings**: `next_embed = soft_token @ embedding_weight`
- **Reward computation**: Forward pass through reward model using soft token distributions
- **KL regularization**: Computed against frozen reference model

### 1.2 GRADE-STE (Straight-Through Estimator)

Same as GRADE but uses hard samples in forward pass with soft gradients in backward:

```python
y_hard = one_hot(argmax(y_soft))
output = (y_hard - y_soft).detach() + y_soft  # STE trick
```

### 1.3 PPO (Proximal Policy Optimization)

Standard PPO implementation with:
- **Clipped surrogate objective** (clip ε = 0.2)
- **Generalized Advantage Estimation** (λ = 0.95, γ = 0.99)
- **Value function** with clipped loss
- **Entropy bonus** for exploration
- **4 optimization epochs** per batch of trajectories

### 1.4 REINFORCE

Vanilla policy gradient baseline:
- **Learned baseline** via exponential moving average (momentum = 0.9)
- **Sequence-level rewards** at end of generation
- **KL penalty** against reference model

---

## 2. Model Architecture

### 2.1 Policy Model

| Component | Configuration |
|-----------|---------------|
| Base Model | `openai-community/gpt2-medium` (default) or `EleutherAI/pythia-410m` |
| Fine-tuning | LoRA adapters |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `query_key_value` (Pythia) or `c_attn, c_proj` (GPT-2) |
| Precision | float32 |

### 2.2 Reward Model

| Component | Configuration |
|-----------|---------------|
| Architecture | Same base model as policy (shared vocabulary) |
| Head | 2-layer MLP classifier (hidden → hidden → 2) |
| Pooling | Mean pooling over sequence |
| Output | Softmax probability of positive class |
| Training | Frozen transformer, only classifier trained |

**Why same vocabulary?** Enables differentiable forward pass—soft token distributions can be converted to soft embeddings via matrix multiplication with shared embedding weights.

### 2.3 Reference Model

- Frozen copy of base model (no LoRA)
- Used for KL divergence computation
- Prevents policy collapse / reward hacking

---

## 3. Data Pipeline

### 3.1 Dataset

**Source**: IMDB Movie Reviews (50,000 total)
- Train split: 25,000 reviews
- Test split: 25,000 reviews
- Labels: Binary sentiment (positive/negative)

### 3.2 Data Splits (No Leakage)

| Split | Size | Source | Purpose |
|-------|------|--------|---------|
| RM Training | 5,000 | IMDB train[0:5000] | Train reward model classifier |
| Policy Training | 10,000 | IMDB train[5000:15000] | Train policy via RL |
| Validation | 2,000 | IMDB train[15000:17000] | Monitor during training |
| Test | 25,000 | IMDB test (full) | Final evaluation only |

**Critical**: Each split is disjoint. The reward model never sees policy training data, and test data is only used once at the end.

### 3.3 Prompt Construction

Reviews are converted to prompts for generation:
1. Take first 1-2 sentences of review
2. Truncate to 200 characters max
3. Tokenize with max length 32, left padding

---

## 4. Training Configuration

### 4.1 Optimization

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 4 |
| Gradient accumulation | 4 steps |
| Effective batch size | 16 |
| Max training steps | 2,000 |
| Gradient clipping | 1.0 (max norm) |

### 4.2 Generation

| Parameter | Value |
|-----------|-------|
| Max new tokens | 48 |
| Min new tokens | 16 |
| Sampling | Top-p = 0.9 |
| Padding side | Left (decoder-only requirement) |

### 4.3 Regularization

| Parameter | Value |
|-----------|-------|
| KL coefficient | 0.1 |
| KL computation | Per-token, summed over sequence |

### 4.4 Method-Specific Parameters

**Gumbel-Softmax (GRADE/STE)**:
| Parameter | Value |
|-----------|-------|
| τ_start | 2.0 |
| τ_end | 0.5 |
| Anneal steps | 1,000 |

**PPO**:
| Parameter | Value |
|-----------|-------|
| PPO epochs | 4 |
| Clip epsilon | 0.2 |
| Value coefficient | 0.5 |
| Entropy coefficient | 0.01 |
| GAE lambda | 0.95 |
| Discount gamma | 0.99 |

**REINFORCE**:
| Parameter | Value |
|-----------|-------|
| Baseline type | Exponential moving average |
| Baseline momentum | 0.9 |

---

## 5. Evaluation Protocol

### 5.1 Validation (During Training)

- **Frequency**: Every 100 steps
- **Samples**: 100 generations
- **Metrics**: Mean reward, std reward
- **Purpose**: Monitor learning, detect overfitting, track best checkpoint

### 5.2 Test (Final Only)

- **When**: After training completes
- **Samples**: 500 generations
- **Metrics**: Mean reward, std reward
- **Purpose**: Unbiased final performance estimate

### 5.3 Tracked Metrics

| Metric | Description |
|--------|-------------|
| `reward` | Training batch reward |
| `loss` | Method-specific loss |
| `kl` | KL divergence from reference |
| `grad_norm_mean` | Mean gradient norm (before clipping) |
| `grad_norm_std` | Gradient norm variance |
| `tau` | Current temperature (GRADE only) |
| `val_reward` | Validation reward |
| `best_val_reward` | Best validation seen |
| `test_reward` | Final test reward |

---

## 6. Infrastructure

| Component | Configuration |
|-----------|---------------|
| Device | CUDA (falls back to CPU) |
| Seed | 42 (reproducibility) |
| Logging | Weights & Biases |
| Output | `./results/{method}/` |

### 6.1 Output Files

Per method:
- `results.json` - All training metrics
- `model/` - Saved LoRA weights

Global:
- `reward_model.pt` - Trained reward model classifier
- `comparison.json` - Summary of all methods

---

## 7. Hypotheses Under Test

1. **Gradient Variance**: GRADE should have lower gradient variance than REINFORCE/PPO due to direct backpropagation vs. sampling-based estimation.

2. **Sample Efficiency**: GRADE should reach target reward levels in fewer steps.

3. **Final Performance**: All methods should achieve similar final rewards (the optimization landscape, not the estimator, determines the optimum).

4. **Generalization**: Methods with lower gradient variance should show smaller val-test gaps.

5. **Training Stability**: GRADE should show smoother learning curves with fewer reward oscillations.

---

## 8. Analysis Pipeline

Post-training analysis via `analysis_script.py`:

1. **Learning curves** - Reward/loss/KL over training
2. **Validation curves** - Val reward + generalization gap
3. **Gradient analysis** - Norm magnitude and variance comparison
4. **Sample efficiency** - Steps to reach reward thresholds
5. **Final comparison** - Test performance bar charts
6. **Temperature ablation** - τ schedule effect on GRADE
7. **Statistical tests** - t-tests for significance
8. **LaTeX table** - Publication-ready results

---

## 9. Running the Experiment

```bash
# Full experiment (all methods)
python training_grade.py

# Single method
# Modify Arguments in main() or add CLI parsing

# Analysis
python analysis_script.py --results_dir ./results
```

---

## 10. Expected Outcomes

Based on the GRADE hypothesis:

| Metric | GRADE vs PPO | GRADE vs REINFORCE |
|--------|--------------|-------------------|
| Test Reward | Comparable (±5%) | Comparable (±5%) |
| Gradient Variance | Lower (significant) | Much lower |
| Steps to 80% | Fewer | Fewer |
| Training Stability | Higher | Higher |

The key insight is not that GRADE achieves higher rewards, but that it does so with **more stable gradients** and **better sample efficiency**, making it a promising alternative for LLM alignment where compute is expensive.

