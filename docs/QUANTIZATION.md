# Quantization — Q8_0 Theory

LLMs have billions of parameters. TinyLlama 1.1B has ~1.1 billion weights.
Stored as 32-bit floats, that's 4.4 GB just for weights. The GGUF file is
only 1.17 GB — quantization is why.

## Why Quantization

A 32-bit float (f32) stores ~7 significant decimal digits. For inference, we
don't need that precision. 8-bit integers (int8) give ~2 significant digits —
enough to distinguish between "very negative", "negative", "zero", "positive",
"very positive" across a distribution. The trade-off is 4× memory savings with
minor quality loss.

## Q8_0 Block Structure

Q8_0 is a **block-wise** quantization scheme. Weights are processed in blocks
of 32. Each block stores:

```
┌──────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  fp16 d  │ q0  │ q1  │ q2  │ ... │     │     │     │     │ q30 │ q31 │
└──────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
│ 2 bytes  │ 32 × int8 (1 byte each) = 32 bytes                           │
└──────────┴───────────────────────────────────────────────────────────────┘
  Total: 34 bytes for 32 weights = 1.0625 bytes/weight
```

**Components:**

1. **Scale (`d`)**: A float16 value representing the maximum absolute value
   within this block. It's the "loudness" of this group of weights.

2. **Quantized values (`q[i]`)**: Each weight is stored as a signed 8-bit
   integer in range [-127, 127]. During quantization:

   ```
   d = max(abs(w[0..31])) / 127.0     # scale factor
   q[i] = round(w[i] / d)             # quantize to int8
   ```

3. **Dequantization** (what ranx does at load time):

   ```
   result[i] = d × q[i]               # reconstruct approximate float
   ```

## Why 32 Weights Per Block?

Smaller blocks (e.g., 16 weights) would track local weight distribution more
accurately but cost more overhead per block. Larger blocks (e.g., 64 weights)
save more space but lose precision because weights within a block may vary
widely. 32 is a sweet spot for transformer weight matrices where adjacent
weights often have similar magnitude.

## Precision Loss

For a typical weight distribution (mean ≈ 0, std ≈ 0.01):

- f32: 23 bits of mantissa → relative error ~1.2 × 10⁻⁷
- Q8_0: quantized to 127 levels per scale → relative error ~1 / (127 × √3) ≈ 0.45%

The block scale preserves precision where magnitude is large (where quantization
error matters most) and loses precision where magnitude is near zero (where
error is less impactful).

## Why Not Q4_0 or Q2_K?

ranx uses Q8_0 for correctness-first development. Lower bit widths would save
more memory but introduce more approximation. Q8_0 is the gentlest quantization
— the output should be nearly identical to f16 inference. Once the inference
engine is verified, lower-bit support can be added by implementing the
corresponding dequant functions.

## Memory Impact on TinyLlama

| Format | Weight Size (1.1B params) | File Size | Status |
|---|---|---|---|
| f32 | 4.4 GB | ~4.4 GB | Baseline |
| f16 | 2.2 GB | — | Would match training precision |
| Q8_0 | 1.17 GB loaded | 1.17 GB on disk | ✅ What we use |
| f32 (dequantized) | 4.4 GB in RAM | — | What ranx loads to |

**Load-time dequant**: ranx reads the compact Q8_0 file (1.17 GB) from disk
and expands every block to f32 in memory (4.4 GB). This uses more RAM but
simplifies the math ops — all computations are plain f32 matrix multiplies.
A future optimization would be to quantize the KV cache and/or keep weights
quantized in memory, doing on-the-fly dequant during matmul (as llama.cpp does).

## Other Quantization Types

The GGUF format defines many quantization types. ranx currently supports
only Q8_0, but the type dispatch in `load_weight_f32` makes it easy to add:

| Type | Bytes/Weight | Description |
|---|---|---|
| F32 | 4 | Full float32 |
| F16 | 2 | Half precision |
| Q8_0 | 1 | 8-bit block (us) |
| Q4_0 | 0.5 | 4-bit block |
| Q4_1 | 0.5 | 4-bit block + min |
| Q5_0 | 0.625 | 5-bit block |
| Q5_1 | 0.625 | 5-bit block + min |
| Q8_1 | 1 | 8-bit block, 2 scales |
| Q2_K | ~0.29 | 2-bit K-quant |
| Q3_K | ~0.44 | 3-bit K-quant |
| Q4_K | ~0.55 | 4-bit K-quant (sweet spot) |
| Q5_K | ~0.69 | 5-bit K-quant |
| Q6_K | ~0.83 | 6-bit K-quant |
