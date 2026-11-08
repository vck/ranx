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

// ─── Viterbi tokenizer ──────────────────────────────────────────────────────

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

    // Viterbi: vocab tokens only. Byte-fallback fills gaps after.
    float *best_score = malloc((norm_len + 1) * sizeof(float));
    int   *best_tok   = malloc((norm_len + 1) * sizeof(int));
    int   *best_len   = malloc((norm_len + 1) * sizeof(int));

    for (int i = 0; i <= norm_len; i++) {
        best_score[i] = -1e30f;
        best_tok[i] = -1;
        best_len[i] = 0;
    }
    best_score[0] = 0.0f;

    for (int pos = 0; pos < norm_len; pos++) {
        if (best_score[pos] < -1e20f) continue;

        unsigned char first = norm[pos];
        int count = tok->prefix_counts[first];
        int *p_ids = tok->prefix_ids[first];

        for (int i = 0; i < count; i++) {
            int tid = p_ids[i];
            int tlen = tok->token_lens[tid];
            if (pos + tlen > norm_len) continue;
            if (memcmp(norm + pos, tok->tokens[tid], tlen) != 0) continue;

            // Skip byte-fallback tokens — they're a last resort
            if (tok->has_byte_fallback && tid >= 0 && tid < 256 &&
                tok->byte_fallback[tid] == tid) continue;
            unsigned char *t = (unsigned char *)tok->tokens[tid];
            if (tlen == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' &&
                is_hex(t[3]) && is_hex(t[4]) && t[5] == '>') continue;

            int npos = pos + tlen;
            float score = (tok->scores && tid < tok->n_tokens) ? tok->scores[tid] : -5.0f;
            // Length bonus: longer tokens strongly preferred
            score += (float)tlen * 0.5f;
            float cand = best_score[pos] + score;
            if (cand > best_score[npos]) {
                best_score[npos] = cand;
                best_tok[npos]   = tid;
                best_len[npos]   = tlen;
            }
        }
    }

    // Build output forward, using Viterbi result where available
    // Fill unreachable positions with byte-fallback
    int *ids = malloc(1024 * sizeof(int));
    int n_ids = 0;
    int cap = 1024;

    int cur = 0;
    while (cur < norm_len) {
        // Look ahead for the best token starting at cur
        int best_id = -1;
        int best_span = 1; // default advance
        unsigned char first = norm[cur];

        // Check if Viterbi found a token at position cur+span that originates at cur
        for (int span = 1; span <= norm_len - cur; span++) {
            if (best_tok[cur + span] >= 0 && best_len[cur + span] == span) {
                best_id = best_tok[cur + span];
                best_span = span;
                break;
            }
        }

        if (best_id >= 0) {
            if (n_ids >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
            ids[n_ids++] = best_id;
            cur += best_span;
        } else {
            // No vocab match from cur — use byte-fallback for 1 byte
            if (tok->has_byte_fallback && tok->byte_fallback[first] >= 0) {
                if (n_ids >= cap) { cap *= 2; ids = realloc(ids, cap * sizeof(int)); }
                ids[n_ids++] = tok->byte_fallback[first];
            }
            cur++;
        }
    }

    *out_len = n_ids;

    free(best_score);
    free(best_tok);
    free(best_len);

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
