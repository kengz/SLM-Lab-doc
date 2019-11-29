# Profiling SLM Lab

## ⏱ Profiling Runtime

When developing a feature in SLM Lab, we may want to profile the program to check its performance and runtime, especially since deep RL software is complicated and involves many components.

We recommend Python's built-in `cProfile` and `snakeviz` to profile your program runtime. The example below runs the profiler and visualizes the program runtime broken down hierarchically by components. See an example of the graph: [https://jiffyclub.github.io/snakeviz/\#interpreting-results](https://jiffyclub.github.io/snakeviz/#interpreting-results)

```bash
conda activate lab
pip install snakeviz

# say, to profile A2C on Pong
python -m cProfile -o a2c.prof run_lab.py slm_lab/spec/benchmark/a2c_gae_pong.json a2c_gae_pong train

# then Ctrl+C to kill the process after some time to collect runtime data
# use snakeviz to render graphs
snakeviz a2c.prof

# a browser will open, showing the runtime breakdown
```

