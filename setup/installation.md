# Installation

## System Dependencies

Install swig (required for Box2D environments like LunarLander):

```bash
# macOS
brew install swig

# Linux (Ubuntu/Debian)
sudo apt-get install -y swig
```

## Installing SLM Lab

Clone the repository:

```bash
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab
```

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Install the CLI tool:

```bash
uv tool install --editable .
```

Verify the installation:

```bash
slm-lab --help
```

{% hint style="warning" %}
If `slm-lab` is not found, restart your terminal or add `~/.local/bin` to your PATH.
{% endhint %}

{% hint style="info" %}
**Book readers:** For the exact code from *Foundations of Deep Reinforcement Learning*, use `git checkout v4.1.1`. See the [book instruction page](../publications-and-talks/instruction-for-the-book-+-intro-to-rl-section.md) for details.
{% endhint %}

### Installing uv

If you don't have uv installed, install it first:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

See [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for more options.

### Alternative Installations

#### Google Colab (2024)

User [**@isacciobota**](https://github.com/isacciobota) has created installation instructions for Google Colab. Please find their detailed instructions with examples here:

{% hint style="info" %}
[SLM Lab Colab notebook (2024)](https://github.com/isacciobota/SLMLab-x-GoogleColab)
{% endhint %}

## What's New in v5

SLM Lab v5 is a modernization release:

* **Gymnasium** replaces OpenAI Gym with proper `terminated`/`truncated` handling
* **uv** replaces conda for fast, reliable dependency management
* **Simpler specs** - no more `body` section or array wrappers
* **Environment names updated** - `CartPole-v1`, `ALE/Pong-v5`, `Hopper-v5`

See the [Changelog](../CHANGELOG.md) for full migration details.

## Hardware Requirements

Non-image based environments can run on a laptop. Only image-based environments such as Atari games benefit from a GPU speedup. For these, we recommend 1 GPU and at least 4 CPUs. This can run a single Atari `Trial` consisting of 4 `Sessions`.

For desktop, a reference spec is RTX 3080+ GPU, 4+ CPUs above 3.0 GHz, and 32 GB RAM.

For cloud computing, use instances with modern GPUs (L4, A10G, or similar). SLM Lab integrates with [dstack](https://dstack.ai/) for cloud training - see [Remote Training](../using-slm-lab/remote-training.md).
