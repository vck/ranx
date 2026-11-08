#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stdbool.h>

// TinyLlama / LLaMA model config
typedef struct {
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int dim;
    int ffn_dim;
    int n_vocab;
    int max_seq_len;
    float rope_freq_base;
    float norm_eps;
} llama_config;

// Per-layer weights
typedef struct {
    float *attn_norm;    // [dim]
    float *ffn_norm;     // [dim]
    float *wq;           // [dim, dim]
    float *wk;           // [dim, n_kv_heads * head_dim]
    float *wv;           // [dim, n_kv_heads * head_dim]
    float *wo;           // [dim, dim]
    float *wgate;        // [dim, ffn_dim]
    float *wdown;        // [ffn_dim, dim]
    float *wup;          // [dim, ffn_dim]
} llama_layer_weights;

// Full model
typedef struct {
    llama_config config;

    float *token_embd;   // [n_vocab, dim]

    llama_layer_weights *layers;

    float *output_norm;  // [dim]
    float *output;       // [dim, n_vocab] — may be NULL if shared with token_embd
    bool output_shared;

    // KV cache: [n_layers, n_kv_heads, max_seq_len, head_dim]
    float *kv_cache_k;
    float *kv_cache_v;

    // Scratch buffers
    float *buf_attn;
} llama_model;

// Load model from GGUF context
bool llama_model_load(llama_model *model, void *gguf_ctx_ptr);

// Free model
void llama_model_free(llama_model *model);

// Forward pass for a single token at given position
// logits: [n_vocab] — output logits for this token
// Returns the score for the correct token (negative if not available)
void llama_forward(llama_model *model, int token, int pos, float *logits);

// Sample from logits (greedy: argmax)
int llama_sample_greedy(const float *logits, int n_vocab);

// Reset KV cache (for new sequence)
void llama_kv_cache_reset(llama_model *model);

#endif
