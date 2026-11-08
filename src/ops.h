#ifndef OPS_H
#define OPS_H

#include <stdint.h>
#include <stdbool.h>

// Matmul: y = W * x
// W: [rows, cols], x: [cols], y: [rows]
void matmul_mv(float *y, const float *W, const float *x, int rows, int cols);

// Matmul batch: C = A * B^T (for single-token attention scores)
// scores[i] = sum_j q[j] * k[i][j]  / sqrt(head_dim)
void attention_scores(float *scores, const float *q, const float *K_cache,
                       int n_kv_heads, int n_heads_per_kv, int head_dim, int seq_len);

// Apply attention scores to V cache to get output
void attention_apply(float *output, const float *scores, const float *V_cache,
                      int n_kv_heads, int n_heads_per_kv, int head_dim, int seq_len);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * w
void rmsnorm(float *y, const float *x, const float *weight, int dim, float eps);

// SiLU (swish): y = x * sigmoid(x)
float silu(float x);

// Softmax in-place over n elements
void softmax(float *x, int n);

// Apply RoPE to query or key vectors
void apply_rope(float *q, int pos, int n_heads, int head_dim, float base_freq);

// ─── Sampling ────────────────────────────────────────────────────────────────

// Apply temperature to logits (in-place)
void apply_temperature(float *logits, int n, float temp);

// Apply repetition penalty to logits (in-place)
void apply_rep_penalty(float *logits, int n, float penalty, const int *prev_tokens, int n_prev);

// Sample from logits with top-k and top-p. Returns token id.
// If top_k <= 0, no top-k filtering. If top_p <= 0, no top-p filtering.
// Uses rand() for stochastic sampling.
int sample_token(float *logits, int n, float temp, int top_k, float top_p);

#endif
