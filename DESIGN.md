# ranx — LLM Inference Engine

ranx is an experimental LLM inference engine in C11, targeting Apple Silicon (M1+).
The first goal is to run a small language model (TinyLlama 1.1B Chat) end-to-end on
Apple M-series chips with Accelerate BLAS, using the GGUF model format.

## Project Structure

```
ranx/
├── src/
│   ├── main.c          — CLI entry point, prompt handling, generation loop
│   ├── gguf.c/h        — GGUF v2/v3 file format parser
│   ├── model.c/h       — LLaMA model loader, weight dequantization, forward pass
│   ├── ops.c/h         — Math ops (matmul, attention, RMSNorm, RoPE, Softmax, sampling)
│   └── tokenizer.c/h   — SentencePiece tokenizer (Viterbi/DP encoding)
├── Makefile            — Build (clang, C11, Accelerate, armv8.5-a+fp16)
├── CMakeLists.txt      — Alternative CMake build
└── DESIGN.md           — This file
```

## Data Flow

```
GGUF file (.gguf)
    │
    ▼
  gguf_parse()          — Parse header, metadata KV, tensor index
    │
    ▼
  llama_model_load()    — Read config, dequantize & load weights to f32,
    │                     allocate KV cache and scratch buffers
    ▼
  tokenizer_init()      — Load vocab strings, scores, build prefix lookup
    │
    ▼
  tokenizer_encode()    — Viterbi DP: find optimal token sequence for prompt
    │
    ▼
  llama_forward()       — Per-layer: RMSNorm → QKV proj → RoPE → cache →
    │                     attention scores → softmax → apply → output proj →
    │                     FFN (SiLU-gate, up, down) → residual → next layer
    ▼
  sample_token()        — Temperature scaling → top-k/top-p filtering → random sample
    │
    ▼
  tokenizer_decode()    — Token ID → text string
    │
    ▼
  (loop to llama_forward for next token)
```

## Component Details

### GGUF Parser (`gguf.c/h`)

Supports GGUFv2 and GGUFv3 with uint64 field widths.

- **Header**: magic (4B), version (4B), tensor_count (u64), metadata_kv_count (u64)
- **Metadata**: tagged union values — scalars (int/float/bool), strings, arrays of scalars/strings. Key stored separately from value to prevent overlap corruption.
- **Tensor Index**: name, dimensions, quantized type, file offset
- **Tensor Data**: starts after all tensor info entries, 32-byte aligned
- **Helpers**: `gguf_find_key()`, `gguf_find_tensor()`, `gguf_tensor_data()`
- **Constants**: GGML types (F32, F16, Q4_0..Q8_0), GGUF value types

### Model Loader & Forward Pass (`model.c/h`)

Reads LLaMA-family architecture from GGUF metadata keys (`llama.*`).

- **Config**: block_count, embedding_length, head_count, head_count_kv, feed_forward_length, context_length, rope.freq_base, norm_eps
- **Weight loading**: per-tensor lookup by name (`blk.N.*`), supports F32 (memcpy), F16 (per-element conversion), Q8_0 (block dequant). All weights stored as f32 for simplicity.
- **Q8_0 dequant**: blocks of 32 weights — 1 fp16 scale + 32 int8 values → 32 f32 values. Partial blocks zero-padded.
- **KV Cache**: `[n_layers × n_kv_heads × max_seq_len × head_dim]` for both K and V
- **Forward pass (single token)**: Embeds token ID → loops over all 22 layers. Each layer does RMSNorm → QKV project → RoPE → cache → GQA attention → output project → residual → RMSNorm → SiLU-gated FFN → residual. Final RMSNorm → output projection to vocab logits.
- **Grouped-Query Attention (GQA)**: 8 query heads per KV head (32:4 ratio), with per-head-per-KV-head score computation.

### Math Ops (`ops.c/h`)

- **matmul_mv**: `cblas_sgemv` for matrix-vector multiply via Accelerate
- **attention_scores**: unrolled GQA — each query head scores all KV positions, scaled by `1/sqrt(head_dim)`
- **attention_apply**: weighted sum of V cache values per head
- **rmsnorm**: `x / sqrt(mean(x²) + eps) * weight`
- **silu**: `x * sigmoid(x)` (SwiGLU activation via the `x * sigmoid(x) * gate` pattern)
- **softmax**: max-subtraction for numerical stability
- **apply_rope**: Neox-style rotary position embeddings — rotates pairs of dimensions by `pos * freq` angle
- **apply_temperature**: scale logits by `1/temp`
- **apply_rep_penalty**: divide logits by penalty for recently-seen tokens
- **sample_token**: temperature → softmax → top-k (40) → top-p (0.9) → random sample via `rand()`

### Tokenizer (`tokenizer.c/h`)

SentencePiece unigram tokenizer with Viterbi (dynamic programming) encoding.

- **Prefix lookup**: pre-built `[256]` arrays of token IDs grouped by first byte for fast matching
- **Byte-fallback**: `<0xXX>` tokens detected in vocabulary, used only as last resort for unreachable positions
- **Viterbi DP**: forward pass finds optimal token sequence using token scores from GGUF metadata + length bonus (`0.5 × token_len_bytes`). Byte-fallback tokens excluded from DP — used only for gap-filling on positions with no matching vocab token.
- **Normalization**: UTF-8 input "hello" → prepend `▁` (U+2581) → replace spaces with `▁` → byte sequence for Viterbi
- **Decode**: token ID → raw string, with `<0xXX>` decoded to actual bytes
- **Special tokens**: BOS (`<s>`, ID=1), EOS (`</s>`, ID=2), UNK (`<unk>`, ID=0)

## Build

```sh
make          # Produces ./ranx binary
make clean    # Remove build artifacts
```

Flags:

- `-std=c11` — C11 standard
- `-march=armv8.5-a+fp16` — Apple Silicon with fp16 support
- `-framework Accelerate` — BLAS via Apple's Accelerate
- `-DACCELERATE_NEW_LAPACK` — Newer LAPACK API

## Usage

```sh
./ranx model.gguf "your prompt here"
```

The prompt is wrapped in the chat template before tokenization.

## Current Status

- ✅ GGUF v2/v3 parsing (header, metadata, tensor index)
- ✅ Q8_0 dequantization to f32
- ✅ LLaMA forward pass with GQA, RoPE, RMSNorm, SiLU FFN
- ✅ KV cache with single-token autoregressive generation
- ✅ SentencePiece tokenizer with Viterbi encoding
- ✅ Byte-fallback token handling
- ✅ Sampling: temperature, top-k, top-p, repetition penalty
- ✅ Apple Silicon Accelerate BLAS integration
- ⚠️ Tokenization quality — byte-fallback still used for some subword boundaries
- ⚠️ Output quality — model generates text but quality is limited
- ⬜ Multi-turn conversation (interactive mode)
- ⬜ Performance profiling
- ⬜ KV cache quantization

## Key Design Decisions

1. **All weights in f32**: Q8_0 is dequantized at load time. Simpler code at the cost of 4× memory. Acceptable for 1.1B model (~5.2 GB in f32 vs 1.17 GB quantized).

2. **Single-token forward**: Process one token at a time with full KV cache recomputation. No batching.

3. **GGUF v3 uint64 everywhere**: v3 uses uint64 for all counts and string lengths. The reader always reads uint64, which accidentally works for v2 files on little-endian (the extra 4 zero bytes are consumed as padding from the next field).

4. **Separate `key` field in `gguf_kv`**: prevents value union from overwriting the key pointer when the value is a string or array (which caused segfaults in early versions).

5. **Viterbi without byte-fallback candidates**: byte-fallback tokens excluded from DP to prevent Viterbi from preferring byte-level tokens over subword tokens.

