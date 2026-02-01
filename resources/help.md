# Help ❓

## Common Issues

### Installation

#### "slm-lab: command not found"

The CLI wasn't added to your PATH. Fix with either:

```bash
# Option 1: Add to PATH permanently (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"

# Option 2: Use uv run prefix
uv run slm-lab --help
```

#### Import errors after `git pull`

Dependencies may have changed. Resync:

```bash
git pull
uv sync
```

#### Box2D/swig errors

Install the system dependency:

```bash
# macOS
brew install swig

# Ubuntu/Debian
sudo apt-get install -y swig
```

### GPU Issues

#### "NVIDIA-SMI has failed"

The NVIDIA driver isn't communicating properly. Reinstall using [this guide](https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07).

#### GPU not detected by PyTorch

Check if CUDA is available:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

If it shows `False`, reinstall PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).

#### Force CPU training

If GPU causes issues, force CPU in the spec:

```javascript
"net": {
  "gpu": false
}
```

### Performance Issues

#### Search mode runs slowly

PyTorch's multi-threading can cause race conditions in parallel search. Fix by limiting threads:

```bash
OMP_NUM_THREADS=1 slm-lab run spec.json spec_name search
```

This is documented in:
- [PyTorch issue #3146](https://github.com/pytorch/pytorch/issues/3146)
- [Imitation issue #274](https://github.com/HumanCompatibleAI/imitation/issues/274)

#### Training is slow

Check if you're using GPU when you should be (or shouldn't be):

| Environment | Best Option |
|-------------|-------------|
| CartPole, LunarLander | CPU (GPU overhead isn't worth it) |
| Atari | GPU |
| MuJoCo | CPU or GPU (GPU helps slightly) |

### Process Issues

#### Kill stuck processes

```bash
# Kill by name
pkill -f slm-lab
pkill -f ray
pkill -f Xvfb

# Or use the built-in command
slm-lab run --stop-ray
```

Monitor processes with [glances](https://github.com/nicolargo/glances):

```bash
uv tool install glances
glances
```

### Headless Server Issues

#### "Cannot connect to None" / No display

Remote servers typically have no display. Two options:

**Option 1: Disable rendering**

```bash
RENDER=false slm-lab run spec.json spec_name train
```

Note: This may prevent some Plotly graphs from generating.

**Option 2: Use Xvfb (recommended)**

Install Xvfb and run with virtual display:

```bash
# Install
sudo apt-get install -y xvfb

# Run
xvfb-run -a slm-lab run spec.json spec_name train
```

#### Forward GUI via SSH

For X11 forwarding:

1. Install X11 on server: `sudo apt-get install xorg`
2. Install XQuartz (macOS) or Xming (Windows) on your laptop
3. SSH with `-X` flag: `ssh -X user@server`

### Data Transfer

#### Sync results from remote server

Use HuggingFace integration (recommended):

```bash
# On remote server
source .env
slm-lab push data/ppo_lunar_2026_01_30_221924

# On local machine
slm-lab pull ppo_lunar
```

Or use rsync:

```bash
rsync -avz user@server:/path/to/SLM-Lab/data/ ./data/
```

## Debugging Tips

### Enable verbose logging

```bash
slm-lab run --log-level DEBUG spec.json spec_name dev
```

### Check spec is valid

Specs are validated on load. Common errors:

| Error | Fix |
|-------|-----|
| "spec_name not found in spec_file" | Check the spec name matches a key in the JSON |
| "Unknown algorithm: X" | Algorithm name must match a class in `slm_lab/agent/algorithm/` |
| "TypeError: missing required argument" | Check spec has all required fields |

### Profile performance

```bash
slm-lab run --profile spec.json spec_name train
```

Or use cProfile directly:

```bash
python -m cProfile -o output.prof -c "from slm_lab.main import main; main(['spec.json', 'spec_name', 'train'])"

# Visualize
uv add snakeviz
snakeviz output.prof
```

## Getting More Help

### Search existing issues

[GitHub Issues](https://github.com/kengz/SLM-Lab/issues) - Many problems have been solved

### Report a new issue

Include:
1. SLM Lab version (`git log -1 --format="%H"`)
2. Python version (`python --version`)
3. OS and hardware
4. Complete error traceback
5. Spec file (if relevant)

[Open new issue](https://github.com/kengz/SLM-Lab/issues/new)

### Community

- [r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/) - RL questions
- [PyTorch Forums](https://discuss.pytorch.org/) - PyTorch-specific issues

## FAQ

### What does SLM stand for?

**Strange Loop Machine** - named after Hofstadter's [Gödel, Escher, Bach](https://www.amazon.com/G%C3%B6del-Escher-Bach-Eternal-Golden/dp/0465026567). SLM Lab was created as part of a long-term AI research project inspired by the book's ideas about self-reference and emergence.

### Why use uv instead of pip/conda?

[uv](https://docs.astral.sh/uv/) is faster and more reliable:
- 10-100x faster than pip
- Lock files for reproducibility
- No environment activation needed (`uv run` handles it)

### Can I use SLM Lab with custom environments?

Yes! Any gymnasium-compatible environment works:

```python
import gymnasium as gym
gym.register(id='MyEnv-v1', entry_point='my_module:MyEnv')
```

Then use `"name": "MyEnv-v1"` in your spec.

### How do I cite SLM Lab?

```bibtex
@misc{kenggraesser2017slmlab,
    author = {Keng, Wah Loon and Graesser, Laura},
    title = {SLM Lab},
    year = {2017},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/kengz/SLM-Lab}},
}
```
