# Profiling SLM Lab

## :stopwatch: Profiling Runtime

When developing a feature in SLM Lab, we may want to profile the program to check its performance and runtime, especially since deep RL software is complicated and involves many components.

We recommend Python's built-in `cProfile` and `snakeviz` to profile your program runtime. The example below runs the profiler and visualizes the program runtime broken down hierarchically by components. See an example of the graph: [https://jiffyclub.github.io/snakeviz/#interpreting-results](https://jiffyclub.github.io/snakeviz/#interpreting-results)

```bash
uv add snakeviz

# say, to profile PPO on CartPole
uv run python -m cProfile -o ppo.prof -c "from slm_lab.main import main; main(['slm_lab/spec/benchmark/ppo/ppo_cartpole.json', 'ppo_cartpole', 'train'])"

# or use the --profile flag for built-in profiling
slm-lab run --profile slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train

# then Ctrl+C to kill the process after some time to collect runtime data
# use snakeviz to render graphs
uv run snakeviz ppo.prof

# a browser will open, showing the runtime breakdown
```
