# TensorBoard 📈

[TensorBoard](https://www.tensorflow.org/tensorboard) provides real-time visualization of training metrics. SLM Lab logs data for TensorBoard in **dev mode** only (not train mode) to minimize overhead during full training runs.

## Quick Start

TensorBoard event files are generated only in **dev mode**:

```bash
# Run in dev mode to enable TensorBoard logging
slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole dev

# Start TensorBoard
uv run tensorboard --log_dir=data

# Open in browser
# http://localhost:6006
```

{% hint style="info" %}
**Train mode**: In train mode, SLM Lab generates CSV files and graphs instead of TensorBoard events to reduce overhead. Use the [generated graphs](session-graph.md) for analysis.
{% endhint %}

## What's Logged

| Category | Metrics | Use Case |
|----------|---------|----------|
| **Scalars** | Rewards, loss, learning rate, FPS | Track training progress |
| **Graphs** | Neural network architecture | Verify model structure |
| **Histograms** | Action distributions, weight distributions | Debug policy behavior |

## Viewing Training Progress

### Scalars Tab

Shows training metrics over time:

- **total_reward**: Episode returns
- **total_reward_ma**: Moving average (100 checkpoints)
- **loss**: Training loss components
- **lr**: Learning rate schedule
- **fps**: Training throughput

### Histograms Tab

Reveals distributions that change over training:

![TensorBoard histograms](https://user-images.githubusercontent.com/8209263/66803221-d9bc0980-eed3-11e9-92b8-0e5cd42a6eab.png)

**Action distributions**: For continuous control (e.g., BipedalWalker with 4 actions), you'll see 4 histogram groups showing how action values evolve. As the agent learns, these distributions should shift and narrow.

**Weight distributions**: Model parameters grouped by layer. Healthy training shows gradual, stable changes. Sudden shifts may indicate instability.

## Tips

### Speed Up Loading

TensorBoard can be slow with many experiments. Specify a single run:

```bash
uv run tensorboard --log_dir=data/ppo_lunar_2026_01_30_221924/log
```

### Compare Multiple Runs

Point to the parent directory to overlay runs:

```bash
uv run tensorboard --log_dir=data
```

Use the "Runs" selector in the UI to toggle visibility.

### Remote Access

When training on a remote server:

```bash
# On server
uv run tensorboard --log_dir=data --bind_all

# Or use SSH tunneling
ssh -L 6006:localhost:6006 user@server
# Then open localhost:6006 locally
```

## TensorBoard vs SLM Lab Graphs

| Feature | TensorBoard | SLM Lab Graphs |
|---------|-------------|----------------|
| Real-time | Yes | No (generated at checkpoints) |
| Interactivity | Full zoom/pan | Basic (Plotly HTML) |
| Aggregation | Manual comparison | Automatic trial averaging |
| Publication-ready | Requires export | PNG ready to use |

Use TensorBoard for debugging during training. Use SLM Lab's generated graphs for final results and publications.
