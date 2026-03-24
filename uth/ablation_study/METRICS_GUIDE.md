# Metrics Guide

This experiment tracks **two complementary metrics** to measure layer importance:

## 1. Translation Loss (Cross-Entropy)

**What it measures**: How well the model predicts the next token given the context using teacher forcing.

**Computation**:
- Concatenate prompt + reference translation
- Compute cross-entropy loss on reference tokens only
- Lower loss = better performance

**Characteristics**:
- ✅ Fast to compute (single forward pass)
- ✅ Differentiable and precise
- ⚠️  Doesn't reflect actual generation quality
- ⚠️  Can be misleading when distribution shifts

## 2. BLEU Score

**What it measures**: Actual translation quality by comparing generated text to reference.

**Computation**:
- Generate translation using greedy decoding
- Compare n-gram overlap with reference (1-4 grams)
- Score range: 0-100 (higher = better)

**Characteristics**:
- ✅ Reflects real translation quality
- ✅ Interpretable (can read actual outputs)
- ✅ Better correlation with human judgment
- ⚠️  Slower (requires generation)
- ⚠️  Can be noisy for single sentences

## Why Both Metrics?

Loss and BLEU can diverge:
- **High loss, high BLEU**: Model uncertain but generates good output
- **Low loss, low BLEU**: Model confident but generates poor output

Tracking both gives a complete picture of intervention impact.

## Output Fields

Each result contains:

```json
{
  // Loss metrics
  "baseline_loss": 1.234,        // Loss without intervention
  "intervened_loss": 2.567,      // Loss with noise at layer
  "loss_delta": 1.333,           // Increase in loss (positive = worse)

  // BLEU metrics
  "baseline_bleu": 45.2,         // BLEU without intervention
  "intervened_bleu": 32.1,       // BLEU with noise at layer
  "bleu_delta": 13.1,            // Decrease in BLEU (positive = worse)

  // Translations (for inspection)
  "baseline_translation": "...", // Generated without noise
  "intervened_translation": "...", // Generated with noise
  "reference": "..."             // Ground truth translation
}
```

## Interpreting Results

### Critical Layers
Layers where interventions cause:
- **High loss_delta** = Important for prediction
- **High bleu_delta** = Important for generation quality

### Agreement
- **Loss and BLEU both high**: Layer is critical
- **Loss high, BLEU low**: Layer affects confidence more than quality
- **Loss low, BLEU high**: Layer affects quality more than confidence

### Example Analysis

```python
# Most critical by loss
df.groupby('layer')['loss_delta'].mean()

# Most critical by BLEU
df.groupby('layer')['bleu_delta'].mean()

# Check correlation
df.plot.scatter(x='loss_delta', y='bleu_delta')
```

## Performance Notes

**Runtime Impact**:
- Loss computation: ~0.1s per sample
- BLEU computation (with generation): ~0.5-1s per sample
- Total: ~2-3x slower than loss-only

**With sample_size=15**:
- Loss-only: ~30 min
- Loss + BLEU: ~60-90 min

Worth it for the insights! 🎯
