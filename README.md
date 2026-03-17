# circuit-tracing-prereqs
Working on Building Skills for AI Safety and Circuit Tracing Research

## Files

- `demo_workflow/test_open_router.py` — Minimal example hitting the OpenRouter API via the OpenAI SDK. Sends a single message and prints the response.
- `toy_transformers/Transformer.py` — Hand-written transformer components (embed/unembed, MLP, attention) built with PyTorch and einops for learning purposes.
- `toy_transformers/modular_arithmetic_transformer.py` — Reimplementation of the modular arithmetic toy model from [Nanda 2023](https://arxiv.org/abs/2301.05217), used to practice and verify PyTorch skills.

## Setup

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

This installs all dependencies into a local virtual environment (`.venv`). PyTorch is pulled from the CUDA 12.4 index on non-Mac machines, so GPU support works out of the box on typical Linux GPU servers.

### Run the OpenRouter demo

Then run:

```bash
uv run python demo_workflow/test_open_router.py
```

Or activate the environment first:

```bash
source .venv/bin/activate
python demo_workflow/test_open_router.py
```
