# Profiling SLM Lab

Understanding performance bottlenecks helps optimize training throughput. SLM Lab includes built-in profiling support and works well with Python's standard profiling tools.

## Quick Start

```bash
# Use built-in profiling (recommended)
slm-lab run --profile slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train

# Let it run for a few minutes, then Ctrl+C to stop and save profile data
```

## Profiling Methods

### Method 1: Built-in `--profile` Flag

The simplest approach:

```bash
slm-lab run --profile spec.json spec_name train
```

This wraps the entire run in cProfile and saves results automatically.

### Method 2: Manual cProfile + snakeviz

For more control over profiling:

```bash
# Install visualization tool
uv add snakeviz

# Profile a training run
uv run python -m cProfile -o ppo.prof -c "from slm_lab.main import main; main(['slm_lab/spec/benchmark/ppo/ppo_cartpole.json', 'ppo_cartpole', 'train'])"

# Run for desired duration, then Ctrl+C to collect data

# Visualize results (opens browser)
uv run snakeviz ppo.prof
```

### Method 3: Line-level Profiling

For detailed function analysis:

```bash
uv add line_profiler

# Add @profile decorator to functions you want to profile
# Then run:
uv run kernprof -l -v your_script.py
```

## Reading Profile Results

### snakeviz Visualization

The snakeviz tool shows a hierarchical breakdown of time spent:

- **Icicle/Sunburst view**: Click to drill into function calls
- **Table view**: Sort by cumulative time, number of calls
- **Hover**: See exact timing and call counts

See [snakeviz documentation](https://jiffyclub.github.io/snakeviz/#interpreting-results) for interpretation guide.

### What to Look For

| Bottleneck | Likely Cause | Fix |
|------------|--------------|-----|
| `env.step()` dominates | Slow environment | Use `num_envs` for parallelization |
| `backward()` slow | Large network | Reduce `hid_layers` or use simpler architecture |
| Memory operations | Inefficient sampling | Check `batch_size`, use GPU |
| `train()` too frequent | Over-training | Increase `training_frequency` |

## Common Performance Patterns

### CPU-bound (Classic Control, Box2D)

```
env.step(): 40%
algorithm.train(): 35%
memory.sample(): 15%
other: 10%
```

Optimization: Increase `num_envs` to parallelize environment stepping.

### GPU-bound (Atari, large networks)

```
algorithm.train(): 60%
  └── net.forward(): 25%
  └── loss.backward(): 30%
env.step(): 25%
other: 15%
```

Optimization: Use larger batch sizes to maximize GPU utilization.

## Monitoring Training Speed

### Real-time FPS

Training logs show frames per second:

```
INFO | frame: 100000, fps: 2341.5, ...
```

Target FPS by environment type:

| Environment | Typical FPS (CPU) | Typical FPS (GPU) |
|-------------|-------------------|-------------------|
| CartPole | 5000-10000 | N/A (CPU better) |
| LunarLander | 2000-4000 | N/A (CPU better) |
| Pong (Atari) | 200-400 | 400-800 |
| HalfCheetah | 500-1000 | 600-1200 |

### Using glances for System Monitoring

```bash
uv tool install glances
glances
```

Monitor CPU, memory, and GPU utilization during training to identify resource bottlenecks.

## Memory Profiling

### RAM Usage by Algorithm

Expected RAM usage varies by algorithm and replay buffer size:

| Algorithm | Memory Driver | Typical RAM |
|-----------|---------------|-------------|
| PPO/A2C | Batch size × num_envs | 1-4 GB |
| DQN/DDQN | Replay buffer (1M transitions default) | 4-8 GB |
| SAC | Replay buffer + twin Q-networks | 6-12 GB |

For off-policy algorithms, replay buffer dominates memory. Reduce `memory.max_size` if RAM-constrained.

### GPU VRAM Usage

VRAM depends on network size and batch size:

| Environment | Network | Batch Size | VRAM |
|-------------|---------|------------|------|
| CartPole | MLP [64,64] | 64 | <1 GB |
| LunarLander | MLP [256,256] | 256 | <1 GB |
| Atari | ConvNet + 512fc | 256 | 2-4 GB |
| MuJoCo | MLP [256,256] | 256 | 1-2 GB |

{% hint style="info" %}
For multi-trial search with `gpu: 0.125`, 8 trials share one GPU. Ensure per-trial VRAM fits within 1/8 of total (e.g., 1-2 GB each on 16 GB GPU).
{% endhint %}

### Monitoring Memory

```bash
# Watch RAM usage
watch -n 1 free -h

# Watch GPU memory
watch -n 1 nvidia-smi

# Detailed Python memory profiling
uv add memory_profiler
uv run python -m memory_profiler your_script.py
```

## Tips for Faster Training

1. **Match hardware to environment**: Use CPU for simple envs, GPU for image-based
2. **Tune `num_envs`**: More parallel envs = better throughput (diminishing returns past ~8-16)
3. **Increase `batch_size`** when using GPU to improve utilization
4. **Reduce `training_frequency`** if algorithm trains too often
5. **Use `--log-level WARNING`** to reduce logging overhead for benchmarks
6. **Reduce replay buffer** for memory-constrained systems (`memory.max_size`)
