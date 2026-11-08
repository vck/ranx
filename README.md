# ranx — Experimental LLM Inference Engine

ranx is a from-scratch LLM inference engine written in C11, targeting Apple Silicon
(M1+). It loads GGUF-format LLaMA models, runs them with Accelerate BLAS, and
generates text one token at a time.

## Quick Start

```sh
make
./ranx model.gguf "what is quantum computing"
```

## Documentation

### Project Design

| Document | Description |
|---|---|
| [DESIGN.md](DESIGN.md) | Architecture overview, data flow, component map |
| [docs/LOADER.md](docs/LOADER.md) | Model loader internals — config, weights, forward pass |

### Theory

| Document | Description |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | LLaMA transformer theory — RMSNorm, RoPE, GQA, SwiGLU |
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md) | Q8_0 block quantization — how 1.17 GB becomes 5.2 GB |
| [docs/TOKENIZATION.md](docs/TOKENIZATION.md) | SentencePiece unigram + Viterbi DP decoding |
| [docs/SAMPLING.md](docs/SAMPLING.md) | Temperature, top-k, top-p, repetition penalty |

## Build

```sh
make          # produce ranx binary
make clean    # remove build artifacts
```

Requires: Apple Clang, Accelerate framework, arm64 macOS.

## Status

- Loads TinyLlama 1.1B Chat (GGUF Q8_0, 201 tensors, 32K vocab)
- Tokenizes with SentencePiece Viterbi encoder
- Runs 22-layer LLaMA forward pass with GQA, RoPE, RMSNorm, SwiGLU
- Generates text with configurable sampling (temp, top-k, top-p, rep penalty)
- Output quality is limited — tokenization still has byte-fallback gaps

## License

MIT
