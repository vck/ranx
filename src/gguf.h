#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#define GGUF_MAGIC "GGUF"
#define GGUF_VERSION 3

// --- GGUF value types ---
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type;

// --- GGML tensor types ---
typedef enum {
    GGML_TYPE_F32   = 0,
    GGML_TYPE_F16   = 1,
    GGML_TYPE_Q4_0  = 2,
    GGML_TYPE_Q4_1  = 3,
    GGML_TYPE_Q5_0  = 6,
    GGML_TYPE_Q5_1  = 7,
    GGML_TYPE_Q8_0  = 8,
    GGML_TYPE_Q8_1  = 9,
    GGML_TYPE_Q2_K  = 10,
    GGML_TYPE_Q3_K  = 11,
    GGML_TYPE_Q4_K  = 12,
    GGML_TYPE_Q5_K  = 13,
    GGML_TYPE_Q6_K  = 14,
    GGML_TYPE_Q8_K  = 15,
} ggml_type;

// --- GGUF header ---
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} gguf_header;

// --- GGUF string ---
typedef struct {
    uint64_t len;
    char *data;
} gguf_string;

// --- GGUF value (tagged union) ---
typedef struct {
    gguf_string key;      // metadata key (stored separately from value)
    gguf_value_type type;
    union {
        uint8_t  uint8;
        int8_t   int8;
        uint16_t uint16;
        int16_t  int16;
        uint32_t uint32;
        int32_t  int32;
        float    float32;
        bool     bool_;
        uint64_t uint64;
        int64_t  int64;
        double   float64;
        gguf_string string;
        struct {
            gguf_value_type array_type;
            uint64_t array_len;
            void *array_data; // array of values, each of type array_type
        } array;
    } value;
} gguf_kv;

// --- GGUF tensor info ---
typedef struct {
    gguf_string name;
    uint32_t n_dims;
    uint64_t *dims;
    ggml_type type;
    uint64_t offset; // relative to start of tensor data section
} gguf_tensor_info;

// --- GGUF context ---
typedef struct {
    uint8_t *file_base;
    size_t   file_size;

    gguf_header header;

    gguf_kv   *metadata;
    uint64_t   n_metadata;

    gguf_tensor_info *tensors;
    uint64_t   n_tensors;

    uint64_t tensor_data_offset; // offset from file base where tensor data starts
} gguf_ctx;

// Parse a GGUF file from a buffer
bool gguf_parse(gguf_ctx *ctx, uint8_t *data, size_t size);
void gguf_free(gguf_ctx *ctx);

// Find metadata key
gguf_kv *gguf_find_key(gguf_ctx *ctx, const char *key);

// Get tensor info by name
gguf_tensor_info *gguf_find_tensor(gguf_ctx *ctx, const char *name);

// Get pointer to tensor data
void *gguf_tensor_data(gguf_ctx *ctx, gguf_tensor_info *t);

// Convert f16 to f32
float fp16_to_fp32(uint16_t h);

// Get number of elements in a tensor
uint64_t gguf_tensor_nelements(gguf_tensor_info *t);

// Get the GGML type size in bytes (for dequantized f32)
int ggml_type_size(ggml_type t);

// Get the number of elements per block for quantized types
int ggml_type_block_size(ggml_type t);

#endif
