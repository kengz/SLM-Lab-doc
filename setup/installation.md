# Installation

## System Dependencies

```bash
# uv - fast Python package manager (handles Python 3.10+ automatically)
curl -LsSf https://astral.sh/uv/install.sh | sh

# swig - required for Box2D environments (LunarLander, BipedalWalker)
brew install swig        # macOS
# apt-get install -y swig  # Linux/WSL
```

{% hint style="warning" %}
**Windows:** Use [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install) and follow Linux instructions.
{% endhint %}

## Install SLM Lab

```bash
git clone https://github.com/kengz/SLM-Lab.git && cd SLM-Lab
uv sync
uv tool install --editable .
slm-lab --help
```

{% hint style="info" %}
If `slm-lab` not found: restart terminal, add `~/.local/bin` to PATH, or use `uv run slm-lab`.
{% endhint %}

### Minimal Install (Orchestration Only)

For dispatching remote training without local ML dependencies:

```bash
git clone https://github.com/kengz/SLM-Lab.git && cd SLM-Lab
uv sync --only-group minimal
uv tool install dstack
```

See [Remote Training](../using-slm-lab/remote-training.md) for dstack setup.

## Docker

```bash
docker pull ghcr.io/kengz/slm-lab:latest
docker run -it ghcr.io/kengz/slm-lab:latest uv run slm-lab --help
docker run -v $(pwd)/data:/app/data ghcr.io/kengz/slm-lab:latest \
    uv run slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
```
