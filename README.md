# llm-internals

![CI](https://github.com/AhmadTawil1/llm-internals/actions/workflows/ci.yml/badge.svg)

A decoder-only transformer built from scratch in PyTorch to internalize the modern LLM stack: RoPE, RMSNorm, LoRA/QLoRA, and a hand-rolled KV-cache.

## Status

🚧 In progress — Month 1 build (May 2026).

## What's inside

- **Attention** — scaled dot-product, multi-head, causal mask
- **RoPE** — rotary positional embeddings
- **RMSNorm** — modern replacement for LayerNorm
- **Transformer block** — attention + norm + FFN + residuals
- **GPT model** — embedding + N blocks + LM head
- **LoRA / QLoRA** — parameter-efficient fine-tuning, from scratch
- **KV-cache** — manual implementation with inference benchmarks

## Architecture

_Diagram coming Week 4._

## Results

_Training runs and KV-cache latency benchmarks coming Week 3–4._

## What I learned

_Coming Week 5 — the honest version, including at least one mistake._

## What's next

_Coming Week 5._

## Stack

PyTorch · pytest · ruff · mypy · GitHub Actions · Weights & Biases

## Project structure

```
llm-internals/
├── src/tinygpt/         # core implementation
├── tests/               # unit tests for each module
├── configs/             # YAML run configs
├── scripts/             # benchmarks and utilities
├── docs/                # architecture notes, full results
└── notebooks/           # exploration only
```

## Setup

```bash
git clone https://github.com/AhmadTawil1/llm-internals.git
cd llm-internals
uv sync
uv run pytest
```

## License

MIT