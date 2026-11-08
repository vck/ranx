#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf.h"
#include <stdbool.h>

typedef struct {
    char  **tokens;        // token strings by ID
    int    *token_lens;     // length of each token
    float  *scores;         // token scores (for Viterbi), may be NULL
    int     n_tokens;
    int     eos_id;
    int     bos_id;
    int     unk_id;

    // byte-fallback: byte value -> token ID
    int     byte_fallback[256];
    bool    has_byte_fallback;

    // fast prefix lookup: for each first byte, list of token indices
    int   **prefix_ids;     // [256] arrays of token indices
    int    *prefix_counts;  // [256] number of tokens per first byte
} tokenizer_t;

bool     tokenizer_init(tokenizer_t *tok, gguf_ctx *gguf);
void     tokenizer_free(tokenizer_t *tok);

int     *tokenizer_encode(tokenizer_t *tok, const char *text, int *out_len);

const char *tokenizer_decode(tokenizer_t *tok, int id);

#endif
