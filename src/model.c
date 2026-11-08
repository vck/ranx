#include "model.h"
#include "gguf.h"
#include "ops.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <Accelerate/Accelerate.h>

// --- Config from GGUF metadata ---
static bool read_config(llama_config *cfg, gguf_ctx *gguf) {
    memset(cfg, 0, sizeof(*cfg));

    gguf_kv *arch_kv = gguf_find_key(gguf, "general.architecture");
    if (!arch_kv || arch_kv->type != GGUF_TYPE_STRING) {
        fprintf(stderr, "No general.architecture found\n");
        return false;
    }
    fprintf(stderr, "Architecture: %.*s\n", (int)arch_kv->value.string.len, arch_kv->value.string.data);

    #define GET_INT(key, field) do { \
        gguf_kv *kv = gguf_find_key(gguf, key); \
        if (kv && (kv->type == GGUF_TYPE_INT32 || kv->type == GGUF_TYPE_UINT32)) { \
            cfg->field = (int)kv->value.int32; \
        } else if (kv && kv->type == GGUF_TYPE_UINT64) { \
            cfg->field = (int)kv->value.uint64; \
        } \
    } while(0)

    #define GET_FLOAT(key, field) do { \
        gguf_kv *kv = gguf_find_key(gguf, key); \
        if (kv && kv->type == GGUF_TYPE_FLOAT32) { \
            cfg->field = kv->value.float32; \
        } \
    } while(0)

    GET_INT("llama.block_count", n_layers);
    GET_INT("llama.embedding_length", dim);
    GET_INT("llama.feed_forward_length", ffn_dim);
    GET_INT("llama.attention.head_count", n_heads);
    GET_INT("llama.attention.head_count_kv", n_kv_heads);
    GET_INT("llama.context_length", max_seq_len);
    GET_FLOAT("llama.rope.freq_base", rope_freq_base);
    GET_FLOAT("llama.attention.layer_norm_rms_epsilon", norm_eps);

    // Also read vocab size from tokenizer metadata
    gguf_kv *vocab_kv = gguf_find_key(gguf, "tokenizer.ggml.tokens");
    if (vocab_kv && vocab_kv->type == GGUF_TYPE_ARRAY) {
        cfg->n_vocab = (int)vocab_kv->value.array.array_len;
    } else {
        cfg->n_vocab = 32000; // default
    }

    // Set defaults if missing
    if (cfg->n_kv_heads == 0) cfg->n_kv_heads = cfg->n_heads;
    if (cfg->rope_freq_base == 0.0f) cfg->rope_freq_base = 10000.0f;
    if (cfg->norm_eps == 0.0f) cfg->norm_eps = 1e-5f;
    if (cfg->max_seq_len == 0) cfg->max_seq_len = 2048;

    fprintf(stderr, "Config: layers=%d dim=%d ffn_dim=%d heads=%d kv_heads=%d vocab=%d max_seq=%d\n",
            cfg->n_layers, cfg->dim, cfg->ffn_dim, cfg->n_heads, cfg->n_kv_heads,
            cfg->n_vocab, cfg->max_seq_len);

    return true;
}

// --- Dequantize Q8_0 block (32 weights, 1 fp16 scale, 32 int8 values) ---
static void dequant_q8_0(float *out, const uint8_t *src, uint64_t nelems) {
    uint64_t blocks = nelems / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint8_t *block = src + b * 34;
        float d = fp16_to_fp32(*(const uint16_t *)block);
        const int8_t *qs = (const int8_t *)(block + 2);
        for (int i = 0; i < 32; i++) {
            out[b * 32 + i] = d * qs[i];
        }
    }
    // Handle remaining elements (if any, pad with 0)
    uint64_t rem = nelems % 32;
    if (rem) {
        for (uint64_t i = nelems - rem; i < nelems; i++) {
            out[i] = 0.0f;
        }
    }
}

// --- Load a single weight tensor from GGUF, converting to f32 ---
static float *load_weight_f32(gguf_ctx *gguf, const char *name) {
    gguf_tensor_info *ti = gguf_find_tensor(gguf, name);
    if (!ti) {
        fprintf(stderr, "  [WARN] tensor missing: %s\n", name);
        return NULL;
    }

    uint64_t nelems = gguf_tensor_nelements(ti);
    float *buf = malloc(nelems * sizeof(float));

    if (ti->type == GGML_TYPE_F32) {
        memcpy(buf, gguf_tensor_data(gguf, ti), nelems * sizeof(float));
    } else if (ti->type == GGML_TYPE_F16) {
        uint16_t *src = (uint16_t *)gguf_tensor_data(gguf, ti);
        for (uint64_t i = 0; i < nelems; i++) {
            buf[i] = fp16_to_fp32(src[i]);
        }
    } else if (ti->type == GGML_TYPE_Q8_0) {
        dequant_q8_0(buf, (const uint8_t *)gguf_tensor_data(gguf, ti), nelems);
    } else {
        fprintf(stderr, "  [WARN] unsupported tensor type %d for %s — need dequant\n", ti->type, name);
        free(buf);
        return NULL;
    }

    fprintf(stderr, "  loaded %s (%llu elements, type=%d)\n", name, (unsigned long long)nelems, ti->type);
    return buf;
}

// --- Allocate KV cache ---
static bool alloc_kv_cache(llama_model *model) {
    llama_config *cfg = &model->config;
    int head_dim = cfg->dim / cfg->n_heads;

    size_t kv_size = (size_t)cfg->n_layers * cfg->n_kv_heads * cfg->max_seq_len * head_dim;

    model->kv_cache_k = calloc(kv_size, sizeof(float));
    model->kv_cache_v = calloc(kv_size, sizeof(float));

    if (!model->kv_cache_k || !model->kv_cache_v) {
        fprintf(stderr, "Failed to allocate KV cache\n");
        return false;
    }

    return true;
}

// --- Load all model weights ---
bool llama_model_load(llama_model *model, void *gguf_ctx_ptr) {
    gguf_ctx *gguf = (gguf_ctx *)gguf_ctx_ptr;
    memset(model, 0, sizeof(*model));

    if (!read_config(&model->config, gguf)) {
        return false;
    }

    llama_config *cfg = &model->config;

    // --- Load weights ---
    model->token_embd = load_weight_f32(gguf, "token_embd.weight");

    model->layers = calloc(cfg->n_layers, sizeof(llama_layer_weights));

    for (int i = 0; i < cfg->n_layers; i++) {
        char name[256];
        llama_layer_weights *l = &model->layers[i];

        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        l->attn_norm = load_weight_f32(gguf, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        l->ffn_norm = load_weight_f32(gguf, name);

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
        l->wq = load_weight_f32(gguf, name);

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
        l->wk = load_weight_f32(gguf, name);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
        l->wv = load_weight_f32(gguf, name);

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
        l->wo = load_weight_f32(gguf, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
        l->wgate = load_weight_f32(gguf, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
        l->wdown = load_weight_f32(gguf, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        l->wup = load_weight_f32(gguf, name);
    }

    model->output_norm = load_weight_f32(gguf, "output_norm.weight");
    model->output = load_weight_f32(gguf, "output.weight");

    // Check if output is shared with token_embd
    // They're shared when output.weight tensor is not found
    if (!model->output) {
        model->output = model->token_embd;
        model->output_shared = true;
        fprintf(stderr, "  output.weight not found, sharing with token_embd.weight\n");
    }

    // --- Allocate KV cache ---
    if (!alloc_kv_cache(model)) {
        return false;
    }

    // --- Allocate scratch buffer ---
    int max_buf = cfg->dim > cfg->ffn_dim ? cfg->dim : cfg->ffn_dim;
    model->buf_attn = calloc(max_buf, sizeof(float));
    if (!model->buf_attn) return false;

    fprintf(stderr, "Model loaded successfully!\n");
    return true;
}

void llama_model_free(llama_model *model) {
    llama_config *cfg = &model->config;

    free(model->token_embd);

    if (model->layers) {
        for (int i = 0; i < cfg->n_layers; i++) {
            llama_layer_weights *l = &model->layers[i];
            free(l->attn_norm);
            free(l->ffn_norm);
            free(l->wq);
            free(l->wk);
            free(l->wv);
            free(l->wo);
            free(l->wgate);
            free(l->wdown);
            free(l->wup);
        }
        free(model->layers);
    }

    free(model->output_norm);
    if (!model->output_shared) {
        free(model->output);
    }

    free(model->kv_cache_k);
    free(model->kv_cache_v);
    free(model->buf_attn);

    memset(model, 0, sizeof(*model));
}

void llama_kv_cache_reset(llama_model *model) {
    llama_config *cfg = &model->config;
    int head_dim = cfg->dim / cfg->n_heads;
    size_t kv_size = (size_t)cfg->n_layers * cfg->n_kv_heads * cfg->max_seq_len * head_dim;
    memset(model->kv_cache_k, 0, kv_size * sizeof(float));
    memset(model->kv_cache_v, 0, kv_size * sizeof(float));
}

// --- Forward pass ---
void llama_forward(llama_model *model, int token, int pos, float *logits) {
    llama_config *cfg = &model->config;
    int head_dim = cfg->dim / cfg->n_heads;
    int kv_dim = cfg->n_kv_heads * head_dim;
    int n_head_per_kv = cfg->n_heads / cfg->n_kv_heads;

    // Embedding lookup
    float *x = malloc(cfg->dim * sizeof(float));
    float *x_norm = malloc(cfg->dim * sizeof(float));
    float *h = malloc(cfg->dim * sizeof(float)); // residual stream
    float *q = malloc(cfg->dim * sizeof(float));
    float *k = malloc(kv_dim * sizeof(float));
    float *v = malloc(kv_dim * sizeof(float));
    float *attn_out = malloc(cfg->dim * sizeof(float));
    float *ffn_gate = malloc(cfg->ffn_dim * sizeof(float));
    float *ffn_up = malloc(cfg->ffn_dim * sizeof(float));
    float *ffn_down = malloc(cfg->dim * sizeof(float));
    float *scores = malloc((cfg->n_heads * (pos + 1)) * sizeof(float)); // scores[head][t]

    const float *emb = model->token_embd + (size_t)token * cfg->dim;
    memcpy(x, emb, cfg->dim * sizeof(float));

    for (int layer = 0; layer < cfg->n_layers; layer++) {
        llama_layer_weights *w = &model->layers[layer];

        // --- Attention ---
        // RMSNorm
        rmsnorm(x_norm, x, w->attn_norm, cfg->dim, cfg->norm_eps);

        // Q projection: q[dim] = wq[dim][dim] * x_norm[dim]
        matmul_mv(q, w->wq, x_norm, cfg->dim, cfg->dim);

        // K projection: k[kv_dim] = wk[kv_dim][dim] * x_norm[dim]
        matmul_mv(k, w->wk, x_norm, kv_dim, cfg->dim);

        // V projection: v[kv_dim] = wv[kv_dim][dim] * x_norm[dim]
        matmul_mv(v, w->wv, x_norm, kv_dim, cfg->dim);

        // Apply RoPE
        apply_rope(q, pos, cfg->n_heads, head_dim, cfg->rope_freq_base);
        apply_rope(k, pos, cfg->n_kv_heads, head_dim, cfg->rope_freq_base);

        // Cache K and V
        int n_pos = pos + 1;
        for (int kvh = 0; kvh < cfg->n_kv_heads; kvh++) {
            float *k_dst = model->kv_cache_k + ((size_t)layer * cfg->n_kv_heads + kvh) * cfg->max_seq_len * head_dim + pos * head_dim;
            float *v_dst = model->kv_cache_v + ((size_t)layer * cfg->n_kv_heads + kvh) * cfg->max_seq_len * head_dim + pos * head_dim;
            const float *k_src = k + kvh * head_dim;
            const float *v_src = v + kvh * head_dim;
            memcpy(k_dst, k_src, head_dim * sizeof(float));
            memcpy(v_dst, v_src, head_dim * sizeof(float));
        }

        // Attention scores
        attention_scores(scores, q,
                         model->kv_cache_k + (size_t)layer * cfg->n_kv_heads * cfg->max_seq_len * head_dim,
                         cfg->n_kv_heads, n_head_per_kv, head_dim, n_pos, cfg->max_seq_len);

        // Softmax over time dimension for each head
        for (int h = 0; h < cfg->n_heads; h++) {
            softmax(scores + h * n_pos, n_pos);
        }

        // Apply attention to values
        attention_apply(attn_out, scores,
                        model->kv_cache_v + (size_t)layer * cfg->n_kv_heads * cfg->max_seq_len * head_dim,
                        cfg->n_kv_heads, n_head_per_kv, head_dim, n_pos, cfg->max_seq_len);

        // Output projection
        memcpy(h, x, cfg->dim * sizeof(float)); // save residual
        matmul_mv(x, w->wo, attn_out, cfg->dim, cfg->dim);
        for (int i = 0; i < cfg->dim; i++) {
            x[i] = h[i] + x[i]; // residual add
        }

        // --- FFN ---
        rmsnorm(x_norm, x, w->ffn_norm, cfg->dim, cfg->norm_eps);

        // Gate
        matmul_mv(ffn_gate, w->wgate, x_norm, cfg->ffn_dim, cfg->dim);
        for (int i = 0; i < cfg->ffn_dim; i++) {
            ffn_gate[i] = silu(ffn_gate[i]);
        }

        // Up
        matmul_mv(ffn_up, w->wup, x_norm, cfg->ffn_dim, cfg->dim);

        // Element-wise multiply
        for (int i = 0; i < cfg->ffn_dim; i++) {
            ffn_up[i] = ffn_gate[i] * ffn_up[i];
        }

        // Down projection
        memcpy(h, x, cfg->dim * sizeof(float));
        matmul_mv(ffn_down, w->wdown, ffn_up, cfg->dim, cfg->ffn_dim);
        for (int i = 0; i < cfg->dim; i++) {
            x[i] = h[i] + ffn_down[i];
        }
    }

    // Final RMSNorm
    rmsnorm(x_norm, x, model->output_norm, cfg->dim, cfg->norm_eps);

    // Output projection
    matmul_mv(logits, model->output, x_norm, cfg->n_vocab, cfg->dim);

    // Cleanup
    free(x);
    free(x_norm);
    free(h);
    free(q);
    free(k);
    free(v);
    free(attn_out);
    free(ffn_gate);
    free(ffn_up);
    free(ffn_down);
    free(scores);
}

int llama_sample_greedy(const float *logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}


