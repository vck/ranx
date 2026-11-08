# Post-mortem: Broken tokenization and empty generation

**Date:** 2026-11-08
**Components:** `src/tokenizer.c`, `src/ops.c`, `src/ops.h`, `src/model.c`
**Severity:** High — every prompt was mis-tokenized and generation produced
nothing or gibberish
**Status:** Fixed and verified against `llama-tokenize` / `libllama`

## Summary

Running a normal prompt produced an empty response: the model emitted `</s>`
immediately. Investigation found **two independent bugs** in the pipeline:

1. **Tokenizer:** `tokenizer_encode()` fragmented text into byte-level and
   single-character tokens. The Viterbi output reconstruction was greedy and
   forward-only, and the model's GGUF scores are rank-like rather than
   log-probabilities, so maximizing their sum was invalid.
2. **Forward pass:** attention read the KV cache with the wrong stride
   (`seq_len` instead of `max_seq_len`), so every KV head after the first read
   the wrong memory. This corrupted attention and made generation collapse to
   EOS or gibberish.

Both are fixed. Tokenization now matches `llama-tokenize` exactly, and greedy
generation produces coherent text.

## Impact

- Prompts were tokenized inconsistently with the model's training tokenizer.
- The model received fragmented IDs, and attention was computed from corrupted
  KV data.
- Typical result: immediate `</s>`, or word-salad output.
- The problems were masked by two misleading signals:
  - `decode(encode(text))` still round-tripped because byte-fallback tokens
    decode to the original bytes.
  - The "Top-5" diagnostic printed after sampling was meaningless because
    `sample_token()` mutates `logits` in place.

## Symptom

```sh
./ranx tinyllama-1.1b-chat-v1.0-q8_0.gguf "what's quantum computing"
```

Output after the tokenizer fix but before the attention fix:

```text
Tokenized (19 tokens): 529,29989,1792,29989,29958,13,5816,29915,29879,12101,20602,13,...

Generated: <s>▁<|user|>
what's▁quantum▁computing
<|assistant|>
</s>
```

Reference `llama-cli` on the same model and prompt:

```text
Quantum computing is a type of computing that relies on the principles of
quantum mechanics to perform calculations and solve problems ...
```

## Timeline / Detection

1. Compared token IDs against `llama-tokenize`. The prompt produced 39 tokens
   instead of 18 (excluding BOS), starting with `29871` (`▁`, score `-1e9`)
   followed by byte-fallback tokens.
2. Fixed the tokenizer by replacing the Viterbi encoder with the
   llama.cpp-compatible SentencePiece bigram-merge algorithm.
3. Re-ran generation. Tokenization was now correct, but the model emitted
   `</s>` immediately.
4. Built two diagnostic tools:
   - `logittest`: loads the model with ranx's code and prints top-k logits.
   - `oracle`: loads the same GGUF through Homebrew `libllama` and prints the
     reference top-k logits.
5. The logits disagreed completely. Reference top-1 was `22930 'Quant'`;
   ranx top-1 was `13` (newline), with `what` and `|` nearby.
6. Reviewed the attention code and found the KV cache stride bug.

## Root Cause 1: Tokenizer

### Forward greedy reconstruction

The Viterbi table stored the best token ending at each position:

```c
best_score[npos] = cand;
best_tok[npos]   = tid;
best_len[npos]   = tlen;
```

Reconstruction started at position 0 and took the shortest matching token:

```c
for (int span = 1; span <= norm_len - cur; span++) {
    if (best_tok[cur + span] >= 0 && best_len[cur + span] == span) {
        best_id = best_tok[cur + span];
        best_span = span;
        break;
    }
}
```

`best_tok[pos]` only records the single best predecessor for `pos`; it does not
mean that token lies on the optimal path from `cur`. It also biases toward the
shortest token. The walk emitted `▁` alone instead of `▁<`, diverged from the
optimal path, and then filled the rest with bytes.

### Wrong score semantics

Even with backward backtracking, the result did not match the reference. The
scores in this GGUF are not log-probabilities:

```text
token 260   '▁t'     score = -1
token 261   'er'     score = -2
token 262   'in'     score = -3
token 529   '▁<'     score = -270
token 1792  'user'   score = -1533
token 3186  '▁world' score = -2927
token 15043 '▁Hello' score = -14784
```

They follow `score ≈ -(id - 259)`, with special tokens at `-1000` and a few
tokens such as `▁` pinned to `-1e9`. Maximizing the sum of these values favors
whichever segmentation uses the lowest-ID tokens — exactly the fragmentation
observed (`▁hello` split into `▁h` + `el` + `lo`).

llama.cpp does not sum these scores for SPM. It merges adjacent symbols, always
choosing the highest-scoring pair whose concatenation is in the vocabulary. That
process stays stable even with rank-like scores.

### Tokenizer fix

Replaced the Viterbi encoder with the SPM bigram-merge encoder:

1. Normalize: prepend `▁`, replace ASCII spaces with `▁`.
2. Split the normalized text into UTF-8 code point symbols with `prev`/`next`
   links.
3. Seed a max-heap with every adjacent pair whose concatenation is a vocab
   token, prioritized by token score (ties by left index).
4. Pop the best pair, merge right into left, update links, and push the two new
   neighbor pairs.
5. Emit vocab tokens for surviving symbols; fall back to `<0xXX>` byte tokens
   when a symbol is not a token.

This is the same algorithm llama.cpp uses for SPM vocabularies.

## Root Cause 2: KV cache stride in attention

The KV cache is laid out as:

```text
[n_layers][n_kv_heads][max_seq_len][head_dim]
```

Writes used `max_seq_len` as the per-head stride:

```c
k_dst = kv_cache_k
      + ((layer * n_kv_heads + kvh) * max_seq_len + pos) * head_dim;
```

Reads in `attention_scores()` and `attention_apply()` used `seq_len`:

```c
const float *kt = K_cache + (kv_head * seq_len + t) * head_dim;
const float *vt = V_cache + (kv_head * seq_len + t) * head_dim;
```

`seq_len` is `n_pos`, which grows during prefill. For `kv_head == 0` the
addresses happen to line up. For `kv_head > 0` they point into the wrong
positions — often uninitialized or zero memory. TinyLlama has four KV heads, so
three quarters of attention was reading garbage. The resulting hidden states
were wrong, which made the model collapse to EOS or produce word salad.

### Forward-pass fix

Added an explicit `kv_stride` parameter to `attention_scores()` and
`attention_apply()` and passed `cfg->max_seq_len` from `llama_forward()`. Both
reads now use:

```c
const float *kt = K_cache + (kv_head * kv_stride + t) * head_dim;
```

## False Lead: RoPE

During debugging, RoPE was changed from consecutive-pair rotation to the NeoX
"rotate half" layout. This was wrong for this model and was reverted. llama.cpp
maps `LLM_ARCH_LLAMA` to `LLAMA_ROPE_TYPE_NORM`, which rotates consecutive pairs
`(d, d+1)` with frequency `base^(-d/head_dim)`. The original implementation was
correct:

```c
for (int d = 0; d < head_dim; d += 2) {
    float theta = 1.0f / powf(base_freq, (float)d / (float)head_dim);
    ...
}
```

The real forward-pass bug was the KV stride, not RoPE.

## Fixes Applied

- `src/tokenizer.c`: replaced Viterbi with the SPM score-ordered bigram merge;
  added `spm_heap`, `vocab_lookup()`, `spm_try_add()`, UTF-8 symbol splitting;
  removed the broken byte-fallback skip condition.
- `src/ops.c` / `src/ops.h`: added `kv_stride` to attention read functions.
- `src/model.c`: pass `cfg->max_seq_len` as the attention stride.
- `src/ops.c`: kept LLaMA normal RoPE (consecutive pairs); NeoX experiment
  reverted.

## Validation

Tokenizer:

- Exact token-ID parity with `llama-tokenize` for plain text, punctuation,
  multiple spaces, tabs, newlines, chat template, accented Latin, Japanese,
  emoji, literal `<0x41>`, and long repeated text.
- Example: the chat prompt is now 18 tokens (excluding BOS) starting
  `529, 29989, 1792, 29989, 29958, 13, ...`.

Forward pass:

- `logittest` (ranx code) and `oracle` (`libllama`) now agree on the top-1 next
  token: `22930 'Quant'`.
- Greedy generation now starts with the same tokens as the reference and
  produces coherent text:

```text
Quantum computing is a new field of computer science that combines the
principles of classical computing and quantum mechanics. It is a powerful
combination of the two disciplines ...
```

- Clean build with `-Wall -Wextra -Wpedantic`; no warnings.

## What Went Wrong in the Debugging Process

- Trusted `decode(encode(x)) == x` as proof of correct tokenization.
- Trusted the "Top-5" printout, which is computed from logits already mutated by
  `sample_token()` (softmax, top-k zeroing, renormalization). It does not show
  raw model logits.
- Assumed GGUF tokenizer scores were log-probabilities without checking their
  range or distribution.
- Nearly "fixed" RoPE based on the wrong convention without confirming which
  rope type llama.cpp uses for `LLM_ARCH_LLAMA`.

## Follow-ups

- Add tokenizer regression tests with fixed expected IDs, generated from
  `llama-tokenize`.
- Add a forward-pass regression test: compare top-1/top-k logits against
  `libllama` for a fixed prompt.
- Fix the `Top-5` diagnostic: snapshot logits before `sample_token()` mutates
  them, or make `sample_token()` operate on a copy.
- Update `docs/TOKENIZATION.md` and `DESIGN.md`, which still describe the old
  Viterbi/DP encoder.
- The tokenizer only implements SPM. The repository's `GLM-4.6V-Flash` GGUF uses
  GPT-2/BPE (`tokenizer.ggml.merges`) and is not supported yet.
- Consider replacing the fixed 16 KiB normalization stack buffer in
  `tokenizer_encode()` with a dynamically sized buffer.
