#include "ops.h"
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void matmul_mv(float *y, const float *W, const float *x, int rows, int cols) {
    // y[rows] = W[rows][cols] * x[cols]
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols,
                1.0f, W, cols, x, 1, 0.0f, y, 1);
}

void attention_scores(float *scores, const float *q, const float *K_cache,
                       int n_kv_heads, int n_heads_per_kv, int head_dim, int seq_len)
{
    int n_heads = n_kv_heads * n_heads_per_kv;
    float inv_scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        int kv_head = h / n_heads_per_kv;
        int q_head_idx = h;

        const float *qh = q + q_head_idx * head_dim;
        float *sh = scores + q_head_idx * seq_len;

        // score against each cached position
        for (int t = 0; t < seq_len; t++) {
            const float *kt = K_cache + (kv_head * seq_len + t) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += qh[d] * kt[d];
            }
            sh[t] = dot * inv_scale;
        }
    }
}

void attention_apply(float *output, const float *scores, const float *V_cache,
                      int n_kv_heads, int n_heads_per_kv, int head_dim, int seq_len)
{
    int n_heads = n_kv_heads * n_heads_per_kv;

    for (int h = 0; h < n_heads; h++) {
        int kv_head = h / n_heads_per_kv;
        int q_head_idx = h;

        const float *sh = scores + q_head_idx * seq_len;
        float *oh = output + q_head_idx * head_dim;

        memset(oh, 0, head_dim * sizeof(float));

        for (int t = 0; t < seq_len; t++) {
            const float *vt = V_cache + (kv_head * seq_len + t) * head_dim;
            float score_t = sh[t];
            for (int d = 0; d < head_dim; d++) {
                oh[d] += score_t * vt[d];
            }
        }
    }
}

void rmsnorm(float *y, const float *x, const float *weight, int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) {
        ss += x[i] * x[i];
    }
    float rms = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int i = 0; i < dim; i++) {
        y[i] = x[i] * rms * weight[i];
    }
}

float silu(float x) {
    return x / (1.0f + expf(-x));
}

void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

void apply_rope(float *q, int pos, int n_heads, int head_dim, float base_freq) {
    for (int h = 0; h < n_heads; h++) {
        float *head = q + h * head_dim;
        for (int d = 0; d < head_dim; d += 2) {
            float theta = 1.0f / powf(base_freq, (float)d / (float)head_dim);
            float cos_val = cosf(pos * theta);
            float sin_val = sinf(pos * theta);

            float q0 = head[d];
            float q1 = head[d + 1];
            head[d]     = q0 * cos_val - q1 * sin_val;
            head[d + 1] = q0 * sin_val + q1 * cos_val;
        }
    }
}

// ─── Sampling ────────────────────────────────────────────────────────────────

void apply_temperature(float *logits, int n, float temp) {
    if (temp <= 0.0f) return; // greedy — handled by caller
    float inv_temp = 1.0f / temp;
    for (int i = 0; i < n; i++) {
        logits[i] *= inv_temp;
    }
}

void apply_rep_penalty(float *logits, int n, float penalty, const int *prev_tokens, int n_prev) {
    if (penalty <= 0.0f) return;
    for (int i = 0; i < n_prev; i++) {
        int id = prev_tokens[i];
        if (id >= 0 && id < n) {
            if (logits[id] > 0) {
                logits[id] /= penalty;
            } else {
                logits[id] *= penalty;
            }
        }
    }
}

// Comparison helper for sorting indices by logit value descending
// macOS qsort_r passes thunk as first arg
static int cmp_desc(void *ctx, const void *a, const void *b) {
    const float *vals = (const float *)ctx;
    int ia = *(const int *)a, ib = *(const int *)b;
    if (vals[ia] > vals[ib]) return -1;
    if (vals[ia] < vals[ib]) return  1;
    return 0;
}

int sample_token(float *logits, int n, float temp, int top_k, float top_p) {
    if (temp > 0.0f) apply_temperature(logits, n, temp);

    softmax(logits, n);

    // Build sorted index array (descending probability)
    int *idx = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;

    // qsort_r for descending sort by logits[idx[i]]
    qsort_r(idx, n, sizeof(int), logits, cmp_desc);

    float sum_all = 0.0f;
    for (int i = 0; i < n; i++) sum_all += logits[i];

    int cutoff = n;

    // Top-k: keep only first k
    if (top_k > 0 && top_k < n && top_k < cutoff) cutoff = top_k;

    // Top-p: find smallest set whose cumulative prob >= top_p
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            cum += logits[idx[i]] / sum_all;
            if (cum >= top_p) { cutoff = i + 1; break; }
        }
    }

    // Zero out tokens outside the candidate set
    for (int i = cutoff; i < n; i++) logits[idx[i]] = 0.0f;

    // Renormalize
    float sum = 0.0f;
    for (int i = 0; i < cutoff; i++) sum += logits[idx[i]];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < cutoff; i++) logits[idx[i]] *= inv;
    } else {
        // Fallback: uniform over cutoff or full vocab
        float inv = 1.0f / (float)cutoff;
        for (int i = 0; i < cutoff; i++) logits[idx[i]] = inv;
    }

    // Sample
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        cum += logits[idx[i]];
        if (r < cum) { int ret = idx[i]; free(idx); return ret; }
    }

    int ret = idx[cutoff - 1];
    free(idx);
    return ret;
}
