#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

// ─── helpers ────────────────────────────────────────────────────────────────
static bool is_hex(int c) {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
}
static int hex_val(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return c - 'a' + 10;
}

static void find_byte_fallback(tokenizer_t *tok) {
    for (int i = 0; i < 256; i++) tok->byte_fallback[i] = -1;
    for (int id = 0; id < tok->n_tokens; id++) {
        const char *s = tok->tokens[id];
        int len = tok->token_lens[id];
        if (len == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' &&
            is_hex(s[3]) && is_hex(s[4]) && s[5] == '>') {
            int b = (hex_val(s[3]) << 4) | hex_val(s[4]);
            tok->byte_fallback[b] = id;
        }
    }
    tok->has_byte_fallback = false;
    for (int i = 0; i < 256; i++) {
        if (tok->byte_fallback[i] >= 0) { tok->has_byte_fallback = true; break; }
    }
}

static void build_prefix_lookup(tokenizer_t *tok) {
    tok->prefix_ids    = calloc(256, sizeof(int *));
    tok->prefix_counts = calloc(256, sizeof(int));

    for (int i = 0; i < tok->n_tokens; i++) {
        if (tok->token_lens[i] == 0) continue;
        unsigned char first = (unsigned char)tok->tokens[i][0];
        tok->prefix_counts[first]++;
    }

    for (int b = 0; b < 256; b++) {
        if (tok->prefix_counts[b] > 0)
            tok->prefix_ids[b] = malloc(tok->prefix_counts[b] * sizeof(int));
    }

    int idx[256];
    memset(idx, 0, sizeof(idx));
    for (int i = 0; i < tok->n_tokens; i++) {
        if (tok->token_lens[i] == 0) continue;
        unsigned char first = (unsigned char)tok->tokens[i][0];
        tok->prefix_ids[first][idx[first]++] = i;
    }
}

// ─── SentencePiece (SPM) tokenizer ──────────────────────────────────────────
//
// SentencePiece models are encoded by iteratively merging adjacent symbols.
// Every adjacent pair whose concatenation exists in the vocabulary is a
// candidate; the highest-scoring candidate is merged, then its new neighbours
// become candidates.  This is the same algorithm llama.cpp uses for SPM
// vocabularies.

typedef struct {
    const unsigned char *text;
    int len;
    int prev;
    int next;
} spm_symbol;

typedef struct {
    int   left;
    int   right;
    int   size;
    float score;
} spm_bigram;

typedef struct {
    spm_bigram *data;
    int         size;
    int         cap;
} spm_heap;

static bool bigram_better(const spm_bigram *a, const spm_bigram *b) {
    if (a->score != b->score) return a->score > b->score;
    return a->left < b->left;
}

static void heap_push(spm_heap *h, spm_bigram b) {
    if (h->size == h->cap) {
        h->cap = h->cap ? h->cap * 2 : 64;
        h->data = realloc(h->data, h->cap * sizeof(spm_bigram));
    }
    int i = h->size++;
    h->data[i] = b;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (!bigram_better(&h->data[i], &h->data[parent])) break;
        spm_bigram tmp = h->data[i];
        h->data[i] = h->data[parent];
        h->data[parent] = tmp;
        i = parent;
    }
}

static spm_bigram heap_pop(spm_heap *h) {
    spm_bigram top = h->data[0];
    h->data[0] = h->data[--h->size];
    int i = 0;
    for (;;) {
        int left = 2 * i + 1, right = left + 1, best = i;
        if (left  < h->size && bigram_better(&h->data[left],  &h->data[best])) best = left;
        if (right < h->size && bigram_better(&h->data[right], &h->data[best])) best = right;
        if (best == i) break;
        spm_bigram tmp = h->data[i];
        h->data[i] = h->data[best];
        h->data[best] = tmp;
        i = best;
    }
    return top;
}

static int vocab_lookup(tokenizer_t *tok, const unsigned char *text, int len) {
    if (len <= 0) return -1;
    unsigned char first = text[0];
    int *ids = tok->prefix_ids[first];
    int count = tok->prefix_counts[first];
    int found = -1;
    for (int i = 0; i < count; i++) {
        int tid = ids[i];
        if (tok->token_lens[tid] == len &&
            memcmp(tok->tokens[tid], text, (size_t)len) == 0) {
            found = tid;
        }
    }
    return found;
}

static void spm_try_add(tokenizer_t *tok, spm_symbol *syms, spm_heap *heap,
                        int left, int right) {
    if (left < 0 || right < 0) return;
    int len = syms[left].len + syms[right].len;
    int tid = vocab_lookup(tok, syms[left].text, len);
    if (tid < 0) return;

    spm_bigram b;
    b.left  = left;
    b.right = right;
    b.size  = len;
    b.score = tok->scores ? tok->scores[tid] : 0.0f;
    heap_push(heap, b);
}

static int utf8_char_len(unsigned char c) {
    static const int lookup[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    return lookup[c >> 4];
}

int *tokenizer_encode(tokenizer_t *tok, const char *text, int *out_len) {
    unsigned char norm[16384];
    int norm_len = 0;
    if (norm_len + 3 <= (int)sizeof(norm)) {
        norm[norm_len++] = 0xE2; norm[norm_len++] = 0x96; norm[norm_len++] = 0x81;
    }
    const unsigned char *p = (const unsigned char *)text;
    while (*p && norm_len < (int)sizeof(norm) - 4) {
        if (*p == ' ') {
            norm[norm_len++] = 0xE2; norm[norm_len++] = 0x96; norm[norm_len++] = 0x81;
        } else {
            norm[norm_len++] = *p;
        }
        p++;
    }

    // Split the normalized text into UTF-8 characters (symbols).
    spm_symbol *syms = malloc((norm_len + 1) * sizeof(spm_symbol));
    int n_syms = 0;
    int offs = 0;
    while (offs < norm_len) {
        int char_len = utf8_char_len(norm[offs]);
        if (char_len > norm_len - offs) char_len = norm_len - offs;
        syms[n_syms].text = norm + offs;
        syms[n_syms].len  = char_len;
        syms[n_syms].prev = n_syms - 1;
        syms[n_syms].next = (offs + char_len == norm_len) ? -1 : n_syms + 1;
        offs += char_len;
        n_syms++;
    }

    // Seed the work queue with every adjacent pair that forms a token.
    spm_heap heap = {0};
    for (int i = 1; i < n_syms; i++) spm_try_add(tok, syms, &heap, i - 1, i);

    // Repeatedly merge the highest-scoring pair.
    while (heap.size > 0) {
        spm_bigram bigram = heap_pop(&heap);
        spm_symbol *left  = &syms[bigram.left];
        spm_symbol *right = &syms[bigram.right];

        if (left->len == 0 || right->len == 0 ||
            left->len + right->len != bigram.size) continue;

        left->len += right->len;
        right->len = 0;

        left->next = right->next;
        if (right->next >= 0) syms[right->next].prev = bigram.left;

        spm_try_add(tok, syms, &heap, left->prev, bigram.left);
        spm_try_add(tok, syms, &heap, bigram.left, left->next);
    }

    // Emit tokens for the surviving symbols, falling back to bytes.
    int cap = 64, n_ids = 0;
    int *ids = malloc(cap * sizeof(int));
    for (int i = 0; i != -1; i = syms[i].next) {
        int tid = vocab_lookup(tok, syms[i].text, syms[i].len);
        if (tid >= 0) {
            if (n_ids >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
            ids[n_ids++] = tid;
            continue;
        }
        for (int j = 0; j < syms[i].len; j++) {
            unsigned char byte = syms[i].text[j];
            int bf = tok->has_byte_fallback ? tok->byte_fallback[byte] : -1;
            if (n_ids >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
            ids[n_ids++] = bf >= 0 ? bf : tok->unk_id;
        }
    }

    free(syms);
    free(heap.data);

    *out_len = n_ids;
    return ids;
}

// ─── public API ─────────────────────────────────────────────────────────────

bool tokenizer_init(tokenizer_t *tok, gguf_ctx *gguf) {
    memset(tok, 0, sizeof(*tok));

    gguf_kv *tokens_kv = gguf_find_key(gguf, "tokenizer.ggml.tokens");
    if (!tokens_kv || tokens_kv->type != GGUF_TYPE_ARRAY ||
        tokens_kv->value.array.array_type != GGUF_TYPE_STRING) {
        fprintf(stderr, "Warning: no tokenizer.ggml.tokens found\n");
        return false;
    }

    tok->n_tokens = (int)tokens_kv->value.array.array_len;
    gguf_string *strs = (gguf_string *)tokens_kv->value.array.array_data;

    tok->tokens     = malloc(tok->n_tokens * sizeof(char *));
    tok->token_lens = malloc(tok->n_tokens * sizeof(int));
    tok->scores     = NULL;

    for (int i = 0; i < tok->n_tokens; i++) {
        tok->tokens[i]     = strndup(strs[i].data, strs[i].len);
        tok->token_lens[i] = (int)strs[i].len;
    }

    // Load scores if available
    gguf_kv *scores_kv = gguf_find_key(gguf, "tokenizer.ggml.scores");
    if (scores_kv && scores_kv->type == GGUF_TYPE_ARRAY &&
        scores_kv->value.array.array_type == GGUF_TYPE_FLOAT32) {
        float *raw = (float *)scores_kv->value.array.array_data;
        tok->scores = malloc(tok->n_tokens * sizeof(float));
        int n_scores = (int)scores_kv->value.array.array_len;
        for (int i = 0; i < tok->n_tokens && i < n_scores; i++) {
            tok->scores[i] = raw[i];
        }
        for (int i = n_scores; i < tok->n_tokens; i++) {
            tok->scores[i] = 0.0f;
        }
        fprintf(stderr, "Tokenizer: loaded %d scores\n", n_scores);
    }

    // Default special ids
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->unk_id = 0;

    find_byte_fallback(tok);
    build_prefix_lookup(tok);

    fprintf(stderr, "Tokenizer: %d tokens | byte-fallback=%s | scores=%s\n",
            tok->n_tokens,
            tok->has_byte_fallback ? "yes" : "no",
            tok->scores ? "yes" : "no");
    return true;
}

void tokenizer_free(tokenizer_t *tok) {
    if (tok->tokens) {
        for (int i = 0; i < tok->n_tokens; i++) free(tok->tokens[i]);
        free(tok->tokens);
    }
    free(tok->token_lens);
    free(tok->scores);
    if (tok->prefix_ids) {
        for (int b = 0; b < 256; b++) free(tok->prefix_ids[b]);
        free(tok->prefix_ids);
    }
    free(tok->prefix_counts);
    memset(tok, 0, sizeof(*tok));
}

const char *tokenizer_decode(tokenizer_t *tok, int id) {
    if (id < 0 || id >= tok->n_tokens) return "<UNK>";
    // Decode byte-fallback tokens like <0x55> -> single byte
    if (tok->has_byte_fallback) {
        const char *s = tok->tokens[id];
        int len = tok->token_lens[id];
        if (len == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' &&
            is_hex(s[3]) && is_hex(s[4]) && s[5] == '>') {
            // Return the single byte as a static buffer
            int b = (hex_val(s[3]) << 4) | hex_val(s[4]);
            static char buf[2];
            buf[0] = (char)b;
            buf[1] = 0;
            return buf;
        }
    }
    return tok->tokens[id];
}
