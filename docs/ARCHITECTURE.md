# LLaMA Architecture Theory

ranx implements the LLaMA (Large Language Model Meta AI) architecture, specifically
the TinyLlama 1.1B variant. This document covers the theoretical foundation of
each component.

## Transformer Foundation

LLaMA is a decoder-only transformer. Unlike encoder-decoder models (like the
original Transformer), it processes a sequence left-to-right, with each token
only attending to itself and previous tokens (causal / autoregressive attention).

The core loop for a single token at position `pos`:

```
hidden ← embedding(token)

for each layer:
    hidden ← hidden + attention(rmsnorm(hidden))
    hidden ← hidden + ffn(rmsnorm(hidden))

logits ← output_projection(rmsnorm(hidden))
```

The "residual stream" (`hidden`) carries information through all layers.
Each sub-layer adds its contribution but the original signal always has a
direct path — this prevents vanishing gradients during training and stabilizes
the forward pass.

## RMSNorm — Root Mean Square Normalization

LLaMA uses RMSNorm instead of LayerNorm. Both normalize activations, but
RMSNorm removes the mean-centering step:

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) * weight
```

LayerNorm:
```
LayerNorm(x) = (x - μ) / sqrt(σ² + ε) * weight + bias
```

RMSNorm is simpler (no μ, no bias) and performs similarly in practice.
The epsilon (ε = 1e⁻⁵) prevents division by zero for near-zero inputs.

In ranx (`ops.c:rmsnorm`):
```
ss = sum(x[i]²)
rms = 1 / sqrt(ss / dim + eps)
y[i] = x[i] * rms * weight[i]
```

LLaMA applies RMSNorm *before* each sub-layer ("pre-norm"), not after
("post-norm"). This is the modern convention and stabilizes training.

## RoPE — Rotary Position Embeddings

Transformers are permutation-invariant — they don't inherently know token
order. Position encodings inject positional information.

**RoPE** encodes position by rotating pairs of embedding dimensions. For
position `pos` and dimension pair `(2i, 2i+1)` within a head:

```
θ_i = base_freq^(-2i / head_dim)
cos = cos(pos × θ_i)
sin = sin(pos × θ_i)

x'[2i]   = x[2i] × cos  -  x[2i+1] × sin
x'[2i+1] = x[2i] × sin  +  x[2i+1] × cos
```

This is equivalent to multiplying by a 2D rotation matrix:

```
[x'[2i]  ]   =  [ cos θ  -sin θ ] [ x[2i]   ]
[x'[2i+1]]      [ sin θ   cos θ ] [ x[2i+1] ]
```

**Why this works**: The rotation encodes position such that the dot product
between two token representations depends only on their *relative* position,
not absolute. This is because rotating both by their respective positions
gives: `Q·K = f(q, k, pos_q - pos_k)`. This relative-position property is
crucial for generalization to unseen sequence lengths.

**Neox-style**: LLaMA uses the "Neox" variant where RoPE is applied to the
first `head_dim` dimensions of each head (rather than interleaving across
the full hidden dimension as in the original RoPE paper).

In ranx (`ops.c:apply_rope`): applied independently to Q and K projections
at each layer.

## GQA — Grouped-Query Attention

Standard multi-head attention has the same number of query, key, and value heads.
Multi-query attention (MQA) uses one KV head shared by all query heads.
Grouped-query attention (GQA) is the middle ground.

TinyLlama: 32 query heads, 4 KV heads → grouping ratio of 8.

```
              Q0  Q1 ... Q7  Q8  Q9 ... Q15  Q16 ... Q23  Q24 ... Q31
              │   │        │   │         │    │         │    │        │
              └───┴──┬─────┘   └──┬──┬───┘    └──┬──┬───┘    └───┬────┘
                     │            │  │           │  │            │
                  KV head 0   KV head 1      KV head 2      KV head 3

     n_heads = 32
     n_kv_heads = 4
     n_heads_per_kv = 32 / 4 = 8  (each KV head serves 8 query heads)
```

**Why GQA**: Full multi-head attention requires loading N KV heads per token.
For 32 heads, that's 32× the memory bandwidth of loading 1. GQA with 4 KV
heads is 8× faster on memory-bound generation while retaining most of the
quality of full multi-head attention.

**In the forward pass** (`ops.c:attention_scores`):

```
for each query head h:
    kv_head = h / n_heads_per_kv           # which KV head this query maps to
    for each time step t:
        score[h][t] = q[h] · k[kv_head][t] / sqrt(head_dim)
```

Each query head scores *all* time steps in the KV cache, but only against
*its assigned* KV head. The scores and value reads group by KV head.

## Causal Attention — The Masking

During training, each token must not attend to future tokens. This is enforced
by adding -infinity to the scores of future positions before softmax.

During inference (generation), we process one token at a time. Token at
position `pos` attends to cached keys at positions [0..pos]. Since future
tokens don't exist yet, no explicit mask is needed — we only compute scores
for existing positions in the KV cache. This is implicit causal masking.

## Attention Output

After softmax normalizes scores to a probability distribution, the attention
output for each query head is:

```
output[h] = sum over t of score[h][t] × V[kv_head][t]
```

This is a weighted sum of the cached value vectors. Tokens with higher
attention scores contribute more to the output.

The per-head outputs are concatenated and projected through the output
weight matrix (`wo`) back to the model dimension.

## SwiGLU FFN

LLaMA uses SwiGLU (Swish-gated Linear Unit) instead of the standard ReLU
or GELU activation in the feed-forward network. SwiGLU has three weight
matrices per layer instead of two:

```
gate = SiLU(x @ W_gate)          # [dim] → [ffn_dim]
up   = x @ W_up                  # [dim] → [ffn_dim]
out  = (gate * up) @ W_down      # [ffn_dim] → [dim]
```

where SiLU (Sigmoid Linear Unit) = `x × sigmoid(x)`.

The `*` is element-wise multiplication. The gate acts as a learned soft
switch — it selects which dimensions of the `up` projection pass through.

**Why SwiGLU**: Empirically, SwiGLU outperforms ReLU and GELU for the same
parameter count. The trade-off is an extra weight matrix (3 instead of 2),
so a SwiGLU FFN with `ffn_dim = 8/3 × dim` has roughly the same parameters
as a standard FFN with `ffn_dim = 4 × dim`.

TinyLlama: `dim=2048, ffn_dim=5632`. The ratio `5632/2048 ≈ 2.75` is between
8/3 ≈ 2.67 (matched params) and 4 (standard).

## Full Layer Computation

Putting it all together for one layer in ranx (`model.c:llama_forward`):

```
def layer(x, pos, layer_weights, kv_cache):
    # ── Attention ──
    residual = x
    x = rmsnorm(x, attn_norm)           # pre-norm

    q = Wq @ x                          # query projection [dim→dim]
    k = Wk @ x                          # key   projection [dim→kv_dim]
    v = Wv @ x                          # value projection [dim→kv_dim]

    q = rope(q, pos, n_heads)           # apply RoPE to query
    k = rope(k, pos, n_kv_heads)        # apply RoPE to key

    kv_cache[k_head][pos] = k           # store in cache
    kv_cache[v_head][pos] = v

    attn = attention_scores(q, kv_cache_k, ...)   # GQA scores
    attn = softmax(attn)                           # normalize
    o = attention_apply(attn, kv_cache_v, ...)     # weighted sum

    x = residual + Wo @ o               # output proj + residual

    # ── FFN ──
    residual = x
    x = rmsnorm(x, ffn_norm)            # pre-norm

    gate = silu(Wgate @ x)              # gate path
    up   = Wup @ x                      # up path
    x = gate * up                       # element-wise gating

    x = residual + Wdown @ x            # down proj + residual

    return x
```

## Numerical Details

| Parameter | TinyLlama Value |
|---|---|
| Layers | 22 |
| Hidden dim | 2048 |
| FFN dim | 5632 |
| Query heads | 32 |
| KV heads | 4 |
| Head dim | 64 |
| Vocab | 32000 |
| Max seq len | 2048 |
| Norm eps | 1e⁻⁵ |
| RoPE base freq | 10000.0 |
