# Installation 🛠️

## Prerequisites ✅

### Python Version

SLM Lab requires Python 3.10+. The `uv` package manager handles Python installation automatically.

### Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that replaces pip and conda:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run `source ~/.bashrc` (or `~/.zshrc`).

{% hint style="warning" %}
**Windows users:** SLM Lab is developed and tested on macOS/Linux. For Windows, use [WSL2](https://docs.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux) and follow the Linux instructions.
{% endhint %}

### Install System Dependencies

**swig** is required for Box2D environments (LunarLander, BipedalWalker):

```bash
# macOS
brew install swig

# Ubuntu/Debian/WSL
sudo apt-get install -y swig
```

{% hint style="info" %}
You can skip swig if you only plan to use CartPole, Atari, or MuJoCo environments. It's only needed for Box2D physics simulation.
{% endhint %}

## Installing SLM Lab 📦

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab

# Install Python dependencies
uv sync

# Install the CLI tool globally
uv tool install --editable .

# Verify installation
slm-lab --help
```

You should see the help menu with available commands.

{% hint style="warning" %}
**PATH issues:** If `slm-lab` is not found, either:
- Restart your terminal
- Add `~/.local/bin` to your PATH
- Or use `uv run slm-lab` instead
{% endhint %}

### Minimal Installation (Orchestration Only)

For machines that only dispatch remote training (no local training):

```bash
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab
uv sync --only-group minimal
uv tool install dstack
```

This installs only the CLI and dstack integration, without heavy ML dependencies.

### MuJoCo Setup

MuJoCo environments work out of the box since MuJoCo became free in 2022. No additional setup needed.

```bash
# Test MuJoCo
slm-lab run slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper dev
```

### GPU Setup (Optional)

SLM Lab automatically uses GPU if available. PyTorch's default installation includes CUDA support.

**Verify GPU detection:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

For specific CUDA versions, see [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Alternative Installations 🐳

### Docker

SLM Lab publishes Docker images:

```bash
# Pull the image
docker pull ghcr.io/kengz/slm-lab:latest

# Run a command
docker run -it ghcr.io/kengz/slm-lab:latest slm-lab --help

# Run training (mount data folder for results)
docker run -v $(pwd)/data:/app/data ghcr.io/kengz/slm-lab:latest \
    slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
```

### Google Colab

Community member [@isacciobota](https://github.com/isacciobota) maintains Colab installation instructions:

{% hint style="info" %}
[SLM Lab Colab notebook (2024)](https://github.com/isacciobota/SLMLab-x-GoogleColab)
{% endhint %}

## Book Readers 📖

If you're following *Foundations of Deep Reinforcement Learning*, use the book-compatible version:

```bash
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab

# Checkout the book version
git checkout v4.1.1

# v4 uses conda
./bin/setup
```

{% hint style="warning" %}
**Version differences:**
- v4.1.1 uses conda, OpenAI Gym, and `python run_lab.py`
- v5+ uses uv, Gymnasium, and `slm-lab run`

The algorithms and concepts are identical; only the tooling and environment APIs changed.
{% endhint %}

## Hardware Requirements

| Use Case | Hardware | Notes |
|----------|----------|-------|
| **Learning/Development** | Laptop (no GPU) | CartPole, LunarLander work fine |
| **Atari Training** | GPU recommended | ~4 hours per game on RTX 3060+ |
| **MuJoCo Training** | CPU sufficient | GPU helps but not required |
| **Benchmarking** | Cloud GPU | Use dstack for on-demand GPUs |

{% hint style="info" %}
**No local GPU?** Use [Remote Training with dstack](../using-slm-lab/remote-training.md) to train on cloud GPUs. Fractional GPU sharing makes it cost-effective ($0.39/hr for L4).
{% endhint %}

## Verify Installation ✨

Run the quick test:

```bash
# Should complete in ~30 seconds
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole dev
```

Press `Ctrl+C` after seeing rewards increase. If it works, proceed to [Quick Start](quick-start.md).

## Troubleshooting

### "slm-lab: command not found"

```bash
# Option 1: Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Option 2: Use uv run
uv run slm-lab --help
```

### Import errors

Ensure you're in the SLM-Lab directory and dependencies are installed:

```bash
cd /path/to/SLM-Lab
uv sync
```

### CUDA/GPU issues

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# Force CPU if GPU issues
# Add to spec: "net": {"gpu": false}
```

For more help, see [Help](../resources/help.md) or [open an issue](https://github.com/kengz/SLM-Lab/issues).
