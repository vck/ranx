# Model Loader — `model.c` / `model.h`

The model loader reads a LLaMA-family model from a parsed GGUF context, dequantizes
weights to f32, allocates the KV cache and scratch buffers, and provides the forward
pass function.

## Entry Point

```c
bool llama_model_load(llama_model *model, void *gguf_ctx_ptr);
```

Called from `main.c` after `gguf_parse()` succeeds. The `gguf_ctx` contains the
fully-parsed GGUF file (metadata KV pairs, tensor index, pointer to raw file data).

## Phases

### 1. Config Read (`read_config`)

Scans GGUF metadata keys for LLaMA architecture parameters. Uses the `GET_INT` /
`GET_FLOAT` macros that look up a key by name and support multiple GGUF value types:

| GGUF Key | Field | Description |
|---|---|---|
| `llama.block_count` | `n_layers` | Number of transformer layers (22 for TinyLlama) |
| `llama.embedding_length` | `dim` | Hidden dimension (2048) |
| `llama.feed_forward_length` | `ffn_dim` | FFN intermediate dimension (5632) |
| `llama.attention.head_count` | `n_heads` | Number of query heads (32) |
| `llama.attention.head_count_kv` | `n_kv_heads` | Number of key/value heads (4 for GQA) |
| `llama.context_length` | `max_seq_len` | Maximum sequence length (2048) |
| `llama.rope.freq_base` | `rope_freq_base` | RoPE base frequency (default 10000.0) |
| `llama.attention.layer_norm_rms_epsilon` | `norm_eps` | RMSNorm epsilon (default 1e-5) |
| `tokenizer.ggml.tokens` | `n_vocab` | Vocabulary size (from array length) |

Defaults are applied for missing optional fields.

### 2. Weight Loading (`load_weight_f32`)

Each weight tensor is loaded by name from the GGUF tensor index. Three formats
are supported:

| GGML Type | Handling |
|---|---|
| `F32` (type=0) | Direct `memcpy` from file buffer |
| `F16` (type=1) | Per-element `fp16_to_fp32()` conversion loop |
| `Q8_0` (type=8) | Block dequantization — see below |

#### Q8_0 Dequantization

Q8_0 packs 32 weights into 34 bytes:
```
[ fp16 scale ][ int8 q0 ][ int8 q1 ] ... [ int8 q31 ]
                  │                           │
                  32 int8 quantized values ────┘
```

Dequant: `out[i] = scale * q[i]` for each 32-weight block. Partial blocks
(the remainder when `nelems % 32 != 0`) are zero-padded.

This runs at load time, so all weights in memory are f32. For TinyLlama 1.1B
this means ~5.2 GB of f32 weights loaded from the 1.17 GB GGUF file.

#### Tensor Names

Each layer `i` loads 9 tensors:

| Name | Shape | Content |
|---|---|---|
| `blk.{i}.attn_norm.weight` | [dim] | Attention RMSNorm scale |
| `blk.{i}.ffn_norm.weight` | [dim] | FFN RMSNorm scale |
| `blk.{i}.attn_q.weight` | [dim, dim] | Query projection (all heads) |
| `blk.{i}.attn_k.weight` | [dim, kv_dim] | Key projection (KV heads only) |
| `blk.{i}.attn_v.weight` | [dim, kv_dim] | Value projection (KV heads only) |
| `blk.{i}.attn_output.weight` | [dim, dim] | Attention output projection |
| `blk.{i}.ffn_gate.weight` | [dim, ffn_dim] | SwiGLU gate projection |
| `blk.{i}.ffn_down.weight` | [ffn_dim, dim] | SwiGLU down projection |
| `blk.{i}.ffn_up.weight` | [dim, ffn_dim] | SwiGLU up projection |

Plus global tensors:

| Name | Shape | Content |
|---|---|---|
| `token_embd.weight` | [n_vocab, dim] | Token embedding table |
| `output_norm.weight` | [dim] | Final RMSNorm scale |
| `output.weight` | [dim, n_vocab] | Output / unembedding (may share with token_embd) |

### 3. KV Cache Allocation

```
K/V cache: [n_layers][n_kv_heads][max_seq_len][head_dim]
                                      │
                              2048 tokens per
                              KV head
```

Where `head_dim = dim / n_heads` (2048 / 32 = 64).

Total size per cache: `22 × 4 × 2048 × 64 = 11,534,336` floats (2 × 44 MB).

The KV cache holds the key and value vectors for every token position in the
current sequence, per layer, per KV head. Cache entries are written at position
`pos` during the forward pass and read back by the attention computation for all
positions ≤ `pos`.

### 4. Scratch Buffers

One reusable buffer of size `max(dim, ffn_dim)` = 5632 floats (~22 KB) for
intermediate values during the forward pass.

## Forward Pass (`llama_forward`)

Signature:

```c
void llama_forward(llama_model *model, int token, int pos, float *logits);
```

Processes one token at position `pos` in the sequence, writing output logits
to `logits[n_vocab]`.

### Per-Layer Operations

```
x = embedding[token]

for layer in 0..n_layers-1:
    # ── Attention sub-block ──
    x_norm = rmsnorm(x, attn_norm[layer])
    q = wq[layer] @ x_norm          # [dim → dim]
    k = wk[layer] @ x_norm          # [dim → kv_dim]
    v = wv[layer] @ x_norm          # [dim → kv_dim]

    q = rope(q, pos, n_heads)        # Neox-style RoPE
    k = rope(k, pos, n_kv_heads)

    kv_cache_k[layer][:, pos] = k    # store KV
    kv_cache_v[layer][:, pos] = v

    # GQA: each query head scores its assigned KV head
    scores[head][t] = q[head] @ kv_cache_k[head//n_head_per_kv][t] / sqrt(head_dim)

    attn = softmax(scores)           # per head, over sequence dim
    o = attention_apply(attn, kv_cache_v)  # weighted sum of V

    x = x + wo[layer] @ o            # output proj + residual

    # ── FFN sub-block ──
    x_norm = rmsnorm(x, ffn_norm[layer])

    gate = silu(wgate[layer] @ x_norm)
    up   = wup[layer] @ x_norm
    x = x + wdown[layer] @ (gate * up)  # residual

# ── Output ──
x_norm = rmsnorm(x, output_norm)
logits = output @ x_norm             # [dim → n_vocab]
```

### Attention Dimensions

| Parameter | Value |
|---|---|
| `dim` (model) | 2048 |
| `n_heads` (query) | 32 |
| `n_kv_heads` | 4 |
| `head_dim` | 2048 / 32 = 64 |
| `kv_dim` | 4 × 64 = 256 |
| `n_head_per_kv` (GQA ratio) | 32 / 4 = 8 |

### RoPE (Neox-style)

Rotates pairs of elements `(x[2i], x[2i+1])` within each head by:

```
angle = pos / freq_base^(2i / head_dim)
cos, sin = cos(angle), sin(angle)
x[2i]   = x[2i] * cos - x[2i+1] * sin
x[2i+1] = x[2i] * sin + x[2i+1] * cos
```

Applied independently to each query and key head.

## Memory Layout

```
┌─────────────────────────────────────────────┐
│              mmap'ed GGUF file               │
│  [header][metadata][tensor index][weights]   │
└─────────────────────────────────────────────┘
         ▲                 ▲
         │                 └── gguf_tensor_data() points here
         │                     (Q8_0 / F16 / F32 raw bytes)
         │
         └── gguf_parse() metadata + index

┌─────────────────────────────────────────────┐
│         Heap-allocated model (f32)           │
│                                              │
│  token_embd:     n_vocab × dim  = 65.5M f32 │
│  layers[0..21]:                              │
│    attn_norm:    dim             =  2K f32   │
│    ffn_norm:     dim             =  2K f32   │
│    wq:           dim × dim       =  4.2M f32 │
│    wk:           dim × kv_dim    =  0.5M f32 │
│    wv:           dim × kv_dim    =  0.5M f32 │
│    wo:           dim × dim       =  4.2M f32 │
│    wgate:        dim × ffn_dim   = 11.5M f32 │
│    wdown:        ffn_dim × dim   = 11.5M f32 │
│    wup:          dim × ffn_dim   = 11.5M f32 │
│                     per layer    ≈ 44.0M f32 │
│                     × 22 layers  ≈ 968.0M f32│
│  output_norm:    dim             =  2K f32   │
│  output:         dim × n_vocab   = 65.5M f32 │
│                                              │
│  Total weights:              ≈ 1.3B f32      │
│  (~5.2 GB @ 4 bytes each)                    │
│                                              │
│  KV cache K:   22×4×2048×64 = 11.5M f32     │
│  KV cache V:   22×4×2048×64 = 11.5M f32     │
│  Scratch buf:  max(dim,ffn) =  5.6K f32     │
└─────────────────────────────────────────────┘
```

## Free

`llama_model_free()` releases all allocated f32 weight buffers, KV cache, and
scratch space. Handles the shared output/embedding case (does not double-free
when `output_shared` is true).
