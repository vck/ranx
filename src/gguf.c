#include "gguf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// --- f16 <-> f32 conversion ---
float fp16_to_fp32(uint16_t h) {
    // IEEE 754-2008 half-precision (binary16)
    // 1 sign | 5 exponent | 10 mantissa
    uint16_t sign   = (h >> 15) & 1;
    uint16_t exp    = (h >> 10) & 0x1F;
    uint16_t mant   = h & 0x3FF;

    uint32_t f32;

    if (exp == 0) {
        // Subnormal: map to zero (flush)
        f32 = sign << 31;
    } else if (exp == 31) {
        // Infinity or NaN
        f32 = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        // Normal: re-bias exponent from 15 to 127
        f32 = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f32, sizeof(result));
    return result;
}

// --- GGML type info ---
int ggml_type_size(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_0: return sizeof(uint16_t) + 16 / 2; // 2 bytes header + 2 bytes per quant
        case GGML_TYPE_Q4_1: return sizeof(uint16_t) * 2 + 16 / 2;
        case GGML_TYPE_Q5_0: return sizeof(uint16_t) + sizeof(uint16_t) + 16 / 2;
        case GGML_TYPE_Q5_1: return sizeof(uint16_t) + sizeof(uint16_t) + sizeof(uint16_t) + 16 / 2;
        case GGML_TYPE_Q8_0: return sizeof(uint16_t) + 32;
        case GGML_TYPE_Q8_1: return sizeof(uint16_t) * 2 + 16;
        default: return -1;
    }
}

int ggml_type_block_size(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return 1;
        case GGML_TYPE_F16:  return 1;
        case GGML_TYPE_Q4_0: return 32;  // blocks of 32 weights
        case GGML_TYPE_Q4_1: return 32;
        case GGML_TYPE_Q5_0: return 32;
        case GGML_TYPE_Q5_1: return 32;
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q8_1: return 32;
        default: return -1;
    }
}

uint64_t gguf_tensor_nelements(gguf_tensor_info *t) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) {
        n *= t->dims[i];
    }
    return n;
}

// --- Internal helpers ---
static uint8_t read_u8(uint8_t **pp) {
    uint8_t v = **pp;
    *pp += 1;
    return v;
}

static int8_t read_i8(uint8_t **pp) {
    int8_t v = *(int8_t *)(*pp);
    *pp += 1;
    return v;
}

static uint16_t read_u16_le(uint8_t **pp) {
    uint16_t v = *(uint16_t *)(*pp);
    *pp += 2;
    return v;
}

static int16_t read_i16_le(uint8_t **pp) {
    int16_t v = *(int16_t *)(*pp);
    *pp += 2;
    return v;
}

static uint32_t read_u32_le(uint8_t **pp) {
    uint32_t v = *(uint32_t *)(*pp);
    *pp += 4;
    return v;
}

static int32_t read_i32_le(uint8_t **pp) {
    int32_t v = *(int32_t *)(*pp);
    *pp += 4;
    return v;
}

static uint64_t read_u64_le(uint8_t **pp) {
    uint64_t v = *(uint64_t *)(*pp);
    *pp += 8;
    return v;
}

static int64_t read_i64_le(uint8_t **pp) {
    int64_t v = *(int64_t *)(*pp);
    *pp += 8;
    return v;
}

static float read_f32_le(uint8_t **pp) {
    float v;
    memcpy(&v, *pp, 4);
    *pp += 4;
    return v;
}

static double read_f64_le(uint8_t **pp) {
    double v;
    memcpy(&v, *pp, 8);
    *pp += 8;
    return v;
}

static gguf_string read_string(uint8_t **pp) {
    gguf_string s;
    s.len = read_u64_le(pp);
    s.data = (char *)(*pp);
    *pp += s.len;
    return s;
}

// Parse a single metadata value
static void read_kv_value(uint8_t **pp, gguf_kv *kv) {
    switch (kv->type) {
        case GGUF_TYPE_UINT8:   kv->value.uint8   = read_u8(pp);     break;
        case GGUF_TYPE_INT8:    kv->value.int8    = read_i8(pp);     break;
        case GGUF_TYPE_UINT16:  kv->value.uint16  = read_u16_le(pp); break;
        case GGUF_TYPE_INT16:   kv->value.int16   = read_i16_le(pp); break;
        case GGUF_TYPE_UINT32:  kv->value.uint32  = read_u32_le(pp); break;
        case GGUF_TYPE_INT32:   kv->value.int32   = read_i32_le(pp); break;
        case GGUF_TYPE_FLOAT32: kv->value.float32 = read_f32_le(pp); break;
        case GGUF_TYPE_BOOL:    kv->value.bool_   = read_u8(pp) != 0; break;
        case GGUF_TYPE_STRING:  kv->value.string  = read_string(pp); break;
        case GGUF_TYPE_UINT64:  kv->value.uint64  = read_u64_le(pp); break;
        case GGUF_TYPE_INT64:   kv->value.int64   = read_i64_le(pp); break;
        case GGUF_TYPE_FLOAT64: kv->value.float64 = read_f64_le(pp); break;
        case GGUF_TYPE_ARRAY: {
            kv->value.array.array_type = (gguf_value_type)read_u32_le(pp);
            kv->value.array.array_len  = read_u64_le(pp);
            size_t elem_size = 0;
            // Rough size for array elements — we store as raw blob
            // For simplicity, we handle strings and scalars separately
            if (kv->value.array.array_type == GGUF_TYPE_STRING) {
                // Array of strings: each entry has a length prefix
                gguf_string *strs = malloc(kv->value.array.array_len * sizeof(gguf_string));
                for (uint64_t i = 0; i < kv->value.array.array_len; i++) {
                    strs[i] = read_string(pp);
                }
                kv->value.array.array_data = strs;
            } else {
                // Scalar array: just read raw bytes
                switch (kv->value.array.array_type) {
                    case GGUF_TYPE_UINT8:
                    case GGUF_TYPE_INT8:    elem_size = 1; break;
                    case GGUF_TYPE_UINT16:
                    case GGUF_TYPE_INT16:   elem_size = 2; break;
                    case GGUF_TYPE_UINT32:
                    case GGUF_TYPE_INT32:
                    case GGUF_TYPE_FLOAT32: elem_size = 4; break;
                    case GGUF_TYPE_UINT64:
                    case GGUF_TYPE_INT64:
                    case GGUF_TYPE_FLOAT64: elem_size = 8; break;
                    default: elem_size = 4; break;
                }
                size_t total = kv->value.array.array_len * elem_size;
                kv->value.array.array_data = malloc(total);
                memcpy(kv->value.array.array_data, *pp, total);
                *pp += total;
            }
            break;
        }
        default:
            fprintf(stderr, "Unknown GGUF value type: %d\n", kv->type);
            break;
    }
}

bool gguf_parse(gguf_ctx *ctx, uint8_t *data, size_t size) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->file_base = data;
    ctx->file_size = size;

    uint8_t *p = data;

    // Read header
    uint32_t magic = read_u32_le(&p);
    if (magic != *(uint32_t *)GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08X\n", magic);
        return false;
    }
    ctx->header.magic = magic;
    ctx->header.version = read_u32_le(&p);

    if (ctx->header.version != 2 && ctx->header.version != 3) {
        fprintf(stderr, "Unsupported GGUF version: %u (expected 2 or 3)\n", ctx->header.version);
        return false;
    }

    ctx->header.tensor_count    = read_u64_le(&p);
    ctx->header.metadata_kv_count = read_u64_le(&p);

    fprintf(stderr, "GGUF v%u | %llu tensors | %llu metadata keys\n",
            ctx->header.version,
            (unsigned long long)ctx->header.tensor_count,
            (unsigned long long)ctx->header.metadata_kv_count);

    // Read metadata
    ctx->n_metadata = ctx->header.metadata_kv_count;
    ctx->metadata = calloc(ctx->n_metadata, sizeof(gguf_kv));

    for (uint64_t i = 0; i < ctx->n_metadata; i++) {
        // Read key
        ctx->metadata[i].key  = read_string(&p);
        ctx->metadata[i].type = (gguf_value_type)read_u32_le(&p);
        read_kv_value(&p, &ctx->metadata[i]);
    }

    // Read tensor info entries
    ctx->n_tensors = ctx->header.tensor_count;
    ctx->tensors = calloc(ctx->n_tensors, sizeof(gguf_tensor_info));

    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        ctx->tensors[i].name   = read_string(&p);
        ctx->tensors[i].n_dims = read_u32_le(&p);
        ctx->tensors[i].dims   = malloc(ctx->tensors[i].n_dims * sizeof(uint64_t));
        for (uint32_t j = 0; j < ctx->tensors[i].n_dims; j++) {
            ctx->tensors[i].dims[j] = read_u64_le(&p);
        }
        ctx->tensors[i].type   = (ggml_type)read_u32_le(&p);
        ctx->tensors[i].offset = read_u64_le(&p);
    }

    // Tensor data starts after all tensor info entries, aligned to 32 bytes
    uint64_t offset = (uint64_t)(p - data);
    offset = (offset + 31) & ~31;
    ctx->tensor_data_offset = offset;

    fprintf(stderr, "Tensor data at offset: %llu (0x%llx)\n",
            (unsigned long long)offset, (unsigned long long)offset);

    return true;
}

void gguf_free(gguf_ctx *ctx) {
    if (!ctx) return;
    // Free tensor dims
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        free(ctx->tensors[i].dims);
    }
    free(ctx->tensors);

    // Free metadata arrays (strings point into file data, but array blobs are malloc'd)
    for (uint64_t i = 0; i < ctx->n_metadata; i++) {
        if (ctx->metadata[i].type == GGUF_TYPE_ARRAY) {
            if (ctx->metadata[i].value.array.array_type == GGUF_TYPE_STRING) {
                free(ctx->metadata[i].value.array.array_data);
            } else {
                free(ctx->metadata[i].value.array.array_data);
            }
        }
    }
    free(ctx->metadata);
    memset(ctx, 0, sizeof(*ctx));
}

gguf_kv *gguf_find_key(gguf_ctx *ctx, const char *key) {
    for (uint64_t i = 0; i < ctx->n_metadata; i++) {
        if (strncmp(ctx->metadata[i].key.data, key, ctx->metadata[i].key.len) == 0 &&
            strlen(key) == ctx->metadata[i].key.len) {
            return &ctx->metadata[i];
        }
    }
    return NULL;
}

gguf_tensor_info *gguf_find_tensor(gguf_ctx *ctx, const char *name) {
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strncmp(ctx->tensors[i].name.data, name, ctx->tensors[i].name.len) == 0 &&
            strlen(name) == ctx->tensors[i].name.len) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

void *gguf_tensor_data(gguf_ctx *ctx, gguf_tensor_info *t) {
    return ctx->file_base + ctx->tensor_data_offset + t->offset;
}
