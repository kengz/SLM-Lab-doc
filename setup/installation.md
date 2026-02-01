# Installation

## System Dependencies

```bash
# uv - fast Python package manager (handles Python 3.10+ automatically)
curl -LsSf https://astral.sh/uv/install.sh | sh

# swig - required for Box2D environments (LunarLander, BipedalWalker)
brew install swig        # macOS
# apt-get install -y swig  # Linux/WSL
```

{% hint style="info" %}
**Windows:** Use [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install) and follow Linux instructions.
{% endhint %}

## Install SLM Lab

```bash
# Clone
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab

# Install
uv sync
uv tool install --editable .

# Verify
slm-lab --help                                 # list all commands
slm-lab run --help                             # options for run command

# Troubleshoot: if slm-lab not found, use uv run
uv run slm-lab --help
```

### Minimal Install

For [remote training](../using-slm-lab/remote-training.md) in the cloud with [dstack](https://dstack.ai), a minimal installation without ML dependencies is available:

```bash
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab
uv sync --only-group minimal
uv tool install dstack
```

## Docker

For containerized runs without local setup, use the [prebuilt image](https://github.com/kengz/SLM-Lab/pkgs/container/slm-lab):

```bash
docker pull ghcr.io/kengz/slm-lab:latest
docker run -it ghcr.io/kengz/slm-lab:latest uv run slm-lab --help
docker run -v $(pwd)/data:/app/data ghcr.io/kengz/slm-lab:latest \
    uv run slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
```
