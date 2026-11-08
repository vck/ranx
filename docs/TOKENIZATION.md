# Tokenization — SentencePiece Unigram + Viterbi Theory

LLMs don't understand characters or words — they process a fixed vocabulary
of **tokens**. The tokenizer is the bridge between raw text and token IDs.

## Why Subword Tokenization

Character-level: long sequences, no morphology awareness.
Word-level: huge vocabulary, can't handle unknown words.
Subword: balances both — common words are single tokens, rare words are
decomposed into meaningful pieces.

Examples:
- "unbelievable" → "un" + "believe" + "able" (subword decomposition)
- "hello" → "▁hello" (prepended space marker)
- "quantum" → "▁quantum" (single token — common word)
- "xylophone" → "▁xy" + "lo" + "phone" (rare → subwords)

## SentencePiece Unigram Language Model

SentencePiece is a tokenizer that operates on raw UTF-8 bytes — no
language-specific pre-tokenization (like splitting on spaces).

### Normalization

Before tokenization, the text is normalized:

1. Prepend a space marker `▁` (U+2581, bytes `E2 96 81`) to the start —
   this lets the model distinguish "hello" (start of sentence) from "▁hello"
   (mid-sentence word).
2. Replace ASCII spaces with `▁` — all spaces become the space marker.
3. Other NFKC normalization may apply (depending on the SentencePiece model).

In ranx (`tokenizer.c:tokenizer_encode`):
```
normalized = ▁ + text_with_spaces_replaced_by_▁
```

### Unigram Model

The unigram language model assigns a score (log-probability) to each token
in the vocabulary. The tokenizer's job is to split the normalized text into
a sequence of known tokens such that the sum of their scores is maximized.

If `score(t)` is the log-probability of token `t`, then for a sequence
of tokens `t₁, t₂, ..., tₙ` covering text positions `[0, n)`:

```
total_score = Σ score(tᵢ)
```

We want the segmentation that *maximizes* this sum. This is an optimization
problem over all possible ways to split the text into known tokens.

## Viterbi Dynamic Programming

The optimal segmentation can be found with dynamic programming — the Viterbi
algorithm. It's the same algorithm used in Hidden Markov Models (HMMs) and
speech recognition.

### The DP Formulation

Define `best[pos]` = maximum cumulative score achievable covering text
positions [0, pos). (0-indexed, exclusive end.)

Base case: `best[0] = 0` (empty prefix has zero cost).

Recurrence: For each position `pos`, for each vocabulary token that matches
at `pos` with length `len`:

```
candidate = best[pos] + score(token) + length_bonus(token)
if candidate > best[pos + len]:
    best[pos + len] = candidate
    best_token[pos + len] = token_id
    best_len[pos + len] = len
```

After filling the DP table, we backtrack from position `n` (end of text) to
position 0, collecting the tokens that gave the optimal score.

### What ranx Does

The ranx tokenizer (`tokenizer.c`, ~160 lines) implements this with:

1. **Prefix lookup**: A `[256]` array pre-built from the vocabulary. For each
   possible first byte (0-255), we store a list of token IDs whose text starts
   with that byte. This filters the 32K vocabulary down to ~dozens of
   candidates per position.

2. **Length bonus**: Token scores from the GGUF metadata are log-probabilities
   (negative values, closer to 0 = more probable). We add `len × 0.5` to the
   score to prefer longer tokens — this biases the algorithm toward
   subword tokens over character-level splits.

3. **Byte-fallback exclusion**: Tokens with the pattern `<0xXX>` (byte-fallback)
   are excluded from the Viterbi DP. They represent individual bytes and would
   dominate if included (one byte always matches at any position). They're used
   only as a last resort for positions where no vocabulary token matches.

### Byte-Fallback

SentencePiece models include 256 special tokens `<0x00>` through `<0xFF>`
that represent any possible byte value. This guarantees that *any* byte
sequence can be tokenized, even if no vocabulary token matches.

**Two strategies:**

| Strategy | Description | ranx approach |
|---|---|---|
| Hard match | Byte-fallback only when no vocab token matches | ✅ Used here |
| Soft match | Byte-fallback included in Viterbi with penalty | Was used earlier |

The hard-match approach avoids the issue where byte-fallback (which matches
at every position) out-scores subword tokens with slightly negative scores.

### Gap Filling

After the Viterbi DP, some positions may have no matching token (when no
vocabulary token starts at that byte). The algorithm's forward walk:

```
cur = 0
while cur < text_len:
    if best_token[cur + span] for some span exists:
        emit(best_token[cur + span])
        cur += span
    else:
        emit(byte_fallback[text[cur]])   # fall back to single byte
        cur += 1
```

This ensures every byte of input is covered while preferring vocabulary tokens.

## Decoding

`tokenizer_decode()` maps a token ID back to text. Byte-fallback tokens
(`<0x55>`) are decoded as the actual byte (0x55 = 'U'), making the output
human-readable.

## Special Tokens

| Token | ID | Meaning |
|---|---|---|
| `<unk>` | 0 | Unknown — should not appear in well-formed input |
| `<s>` | 1 | Start of sequence (BOS) |
| `</s>` | 2 | End of sequence (EOS) |

The BOS token is prepended to the tokenized prompt before the first forward
pass.

## Tokenization Example

For input "hello" (5 bytes + prepended ▁ = 8 bytes):

```
Normalized: E2 96 81 68 65 6C 6C 6F
              ▁       h  e  l  l  o

Viterbi finds: token 29871 = "▁" (3 bytes, the space marker)
               token 9202  = "hello" (5 bytes)
```

Instead of naive byte-level: `▁` + `h` + `e` + `l` + `l` + `o` (6 tokens),
Viterbi prefers the single subword token `▁hello` (2 tokens) because of the
length bonus and token score.
