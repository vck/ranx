# Sampling Theory

After the forward pass, the model produces a vector of **logits** — one float
per vocabulary token. These are raw scores, not probabilities. Sampling converts
them into a chosen next token.

## The Generation Loop

```
for step = 0..max_tokens:
    logits = model.forward(prompt_tokens ++ generated_tokens)
    logits = temperature(logits)
    probs  = softmax(logits)
    probs  = top_k(probs)
    probs  = top_p(probs)
    probs  = rep_penalty(probs, recent_tokens)
    next_token = sample(probs)
    if next_token == EOS: stop
    emit(next_token)
```

Each step uses the full context (prompt + all previously generated tokens),
made efficient by the KV cache.

## From Logits to Probabilities

A logit `lᵢ` is the raw output of the final linear layer. It can be any real
number — negative, zero, positive. The softmax converts logits to a probability
distribution:

```
pᵢ = exp(lᵢ) / Σⱼ exp(lⱼ)
```

Properties:
- All pᵢ > 0 (every token has non-zero probability)
- Σ pᵢ = 1 (it's a valid distribution)
- Larger lᵢ → larger pᵢ (monotonic)

The ratio between probabilities is exponential: if token A has logit 5.0 and
token B has logit 4.0, p(A) / p(B) = exp(1.0) ≈ 2.7. Small logit differences
become large probability ratios.

## Temperature

Temperature flattens or sharpens the distribution before softmax:

```
l'_i = l_i / T
```

- **T → 0**: The distribution becomes a spike at the max logit (greedy).
  `lim(T→0) softmax(l/T) = one-hot(argmax(l))`
- **T = 1**: No change. The model's native distribution.
- **T → ∞**: The distribution approaches uniform. All tokens equally likely.

**Why use temperature?** Greedy decoding (T=0) produces repetitive, boring
text. Higher temperature increases diversity but risks incoherence. Typical
values: 0.7-0.9 for creative text, 0.3-0.5 for factual answers.

In ranx (`ops.c:apply_temperature`): `logits[i] /= temp`. With temp=0.0,
the function returns early — the caller handles greedy by setting top_k=1.

## Softmax (In Practice)

Naive softmax `exp(x) / sum(exp(x))` overflows for large x (exp(100) ≈
2.7 × 10⁴³). The fix: subtract the maximum value before exponentiating.

```
m = max(x)
p_i = exp(x_i - m) / Σⱼ exp(x_j - m)
```

This is mathematically equivalent (the exp offset cancels in numerator and
denominator) but numerically stable. ranx (`ops.c:softmax`) implements this.

## Top-k Filtering

Keep only the `k` highest-probability tokens and redistribute probability
mass among them. All other tokens get probability zero.

```
sorted_indices = argsort(probs, descending)
cutoff = min(k, vocab_size)
probs[sorted_indices[cutoff:]] = 0
probs /= sum(probs)                          # renormalize
```

**Why top-k?** The model's distribution often has a long tail of very
low-probability tokens. Sampling from the full distribution occasionally
picks nonsensical tokens. Top-k truncates this tail.

Typical `k`: 30-50 for generation. ranx defaults to 40.

**Drawback**: k is fixed. Sometimes the model is very confident (top-3
probability is 0.95) but we still force inclusion of 37 low-probability tokens.
Sometimes the model is confused (top-100 probability is 0.6) and we cut off
viable tokens.

## Top-p (Nucleus) Sampling

Instead of a fixed number of tokens, keep the smallest set whose cumulative
probability exceeds threshold `p`:

```
sorted = argsort(probs, descending)
cumulative = 0
cutoff = vocab_size
for i in 0..vocab_size:
    cumulative += probs[sorted[i]]
    if cumulative >= p:
        cutoff = i + 1
        break
probs[sorted[cutoff:]] = 0
probs /= sum(probs)
```

**Why top-p?** It adapts to the model's confidence:
- Confident distribution: few tokens kept (top-3 or top-5)
- Uncertain distribution: more tokens kept (top-100+)

In practice, top-p often works better than top-k alone. ranx combines both:
top-k first (reduce the pool), then top-p (fine-tune the cutoff).

Typical `p`: 0.9-0.95. ranx defaults to 0.9.

## Repetition Penalty

LLMs tend to repeat tokens or short phrases. The repetition penalty reduces
the probability of recently-generated tokens:

```
for each token in recent_tokens:
    if logits[token] > 0:
        logits[token] /= penalty
    else:
        logits[token] *= penalty
```

The asymmetry matters: positive logits are suppressed (divided), negative
logits are enhanced (multiplied). This discourages repetition without
completely forbidding it.

**How it works**: Divided means the token's probability goes down. But the
penalty applies *before* softmax, so the total distribution is renormalized
— other tokens naturally get slightly higher probability.

**Typical values**: 1.0-1.2. 1.0 = no penalty. 1.1 = mild (ranx default).
1.2 = strong. Above 1.3 often causes incoherence (model avoids obvious words).

ranx applies the penalty over the last 10 generated tokens.

## Putting It All Together

ranx's `sample_token()` (`ops.c:144-199`) executes this pipeline:

1. **Temperature**: scale logits by 1/temp
2. **Softmax**: convert to probabilities
3. **Sort**: descending order by probability (via `qsort_r`)
4. **Top-k**: keep first k entries, zero others
5. **Top-p**: further reduce to smallest set above cumulative threshold p
6. **Renormalize**: ensure probabilities sum to 1
7. **Sample**: draw a token from the filtered distribution via `rand()`

The default parameters (temp=0.7, top_k=40, top_p=0.9, rep_penalty=1.1) are
general-purpose settings suitable for open-ended text generation.

## Greedy Decoding

For deterministic output (useful for debugging or factual tasks), set:

```
temperature = 0.0   # skip temperature scaling
top_k = 1           # keep only the top token
top_p = 1.0         # no top-p filtering
```

This always selects the token with the highest logit — the argmax. No
randomness, reproducible results, but often boring and repetitive.

## Why Sampling Matters for Quality

Without sampling tricks, even a perfectly trained model produces poor output.
The probability distribution at each step contains many plausible continuations.
Picking only the most likely one (`argmax`) leads to:

- **Repetition**: "I like cats. I like dogs. I like birds." (model gets stuck
  in the most likely pattern)
- **Short horizons**: The model can't "plan ahead" — it greedily picks the
  locally-best token, which often leads to dead ends.
- **Lack of creativity**: The most likely next word is often the most boring.

Stochastic sampling with temperature, top-k, and top-p mimics the diversity of
human language by occasionally picking the 2nd or 10th most likely token,
which then leads to a completely different (and often better) continuation.
