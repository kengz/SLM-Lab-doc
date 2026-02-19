# TorchArc

## Declarative YAML Network Architectures

Code: [slm\_lab/agent/net/torcharc\_net.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/net/torcharc_net.py) · Library: [torcharc](https://github.com/kengz/torcharc)

TorchArc builds neural networks from declarative YAML specs via the [torcharc](https://github.com/kengz/torcharc) library. Instead of hardcoded network classes like `MLPNet` or `ConvNet`, you define the exact PyTorch modules and dataflow graph in YAML. All `benchmark_arc/` specs use TorchArc (v5.1+).

## YAML Structure

A TorchArc spec has two parts: **modules** (what layers to build) and **graph** (how data flows through them).

```yaml
net:
  type: TorchArcNet
  shared: false
  arc:
    modules:
      body:
        Sequential:
          - LazyLinear: {out_features: 256}
          - ReLU:
          - LazyLinear: {out_features: 256}
          - ReLU:
    graph:
      input: x
      modules:
        body: [x]
      output: body
  hid_layers_activation: relu
  clip_grad_val: 0.5
  optim_spec:
    name: Adam
    lr: 3.0e-4
  gpu: auto
```

- **`modules`** — named groups of PyTorch modules. Each key (e.g. `body`) becomes a callable sub-network. Use any `torch.nn` module by name.
- **`graph`** — defines the input name, which modules receive which inputs, and which module produces the final output.
- `LazyLinear` / `LazyConv2d` — input dimensions are inferred automatically, so you only specify output sizes.

## YAML Anchors for Compact Specs

Real benchmark specs use YAML anchors (`&name` / `*name`) to define a base config once and reuse it across environments. This is the key to keeping multi-environment specs DRY.

### Define once, reuse everywhere

```yaml
# Define the base architecture once
_ppo_mujoco_arc: &ppo_mujoco_arc
  modules:
    body:
      Sequential:
        - LazyLinear: {out_features: 256}
        - Tanh:
        - LazyLinear: {out_features: 256}
        - Tanh:
  graph:
    input: x
    modules:
      body: [x]
    output: body

# Define the base net config once
_ppo_mujoco_net: &ppo_mujoco_net
  type: TorchArcNet
  shared: false
  arc: *ppo_mujoco_arc          # ← reference the architecture
  hid_layers_activation: tanh
  init_fn: orthogonal_
  clip_grad_val: 0.5
  use_same_optim: true
  optim_spec:
    name: AdamW
    lr: 3.0e-4
  gpu: auto
```

### Merge and override per environment

Use `<<: *anchor` to inherit all fields, then override only what differs:

```yaml
ppo_ant_arc:
  agent:
    net:
      <<: *ppo_mujoco_net        # ← inherit everything
      optim_spec:                 # ← override just the lr
        name: AdamW
        lr: 1.5e-4

ppo_hopper_arc:
  agent:
    net:
      <<: *ppo_mujoco_net        # ← same base
      lr_scheduler_spec:          # ← add lr decay for this env
        name: LinearToZero
        frame: "${max_frame}"
```

{% hint style="info" %}
**Why this matters:** Each environment's spec shows _only its differences_ from the base. When scanning `ppo_mujoco_arc.yaml`, you immediately see that Ant uses a lower learning rate while Hopper adds LR scheduling—without wading through identical boilerplate.
{% endhint %}

### Before vs. after

<details>
<summary><b>Without anchors</b> — repetitive, hard to diff</summary>

```yaml
ppo_ant_arc:
  agent:
    net:
      type: TorchArcNet
      shared: false
      arc:
        modules:
          body:
            Sequential:
              - LazyLinear: {out_features: 256}
              - Tanh:
              - LazyLinear: {out_features: 256}
              - Tanh:
        graph:
          input: x
          modules:
            body: [x]
          output: body
      hid_layers_activation: tanh
      init_fn: orthogonal_
      clip_grad_val: 0.5
      use_same_optim: true
      optim_spec:
        name: AdamW
        lr: 1.5e-4        # ← the only actual difference
      gpu: auto

ppo_hopper_arc:
  agent:
    net:
      type: TorchArcNet
      shared: false
      arc:
        modules:
          body:
            Sequential:
              - LazyLinear: {out_features: 256}
              - Tanh:
              - LazyLinear: {out_features: 256}
              - Tanh:
        graph:
          input: x
          modules:
            body: [x]
          output: body
      hid_layers_activation: tanh
      init_fn: orthogonal_
      clip_grad_val: 0.5
      use_same_optim: true
      optim_spec:
        name: AdamW
        lr: 3.0e-4
      lr_scheduler_spec:   # ← the only actual difference
        name: LinearToZero
        frame: "${max_frame}"
      gpu: auto
```

</details>

**With anchors** — only overrides are visible:

```yaml
ppo_ant_arc:
  agent:
    net:
      <<: *ppo_mujoco_net
      optim_spec: {name: AdamW, lr: 1.5e-4}

ppo_hopper_arc:
  agent:
    net:
      <<: *ppo_mujoco_net
      lr_scheduler_spec: {name: LinearToZero, frame: "${max_frame}"}
```

## Atari (Conv) Architecture

TorchArc handles convolutional networks the same way—list the modules explicitly:

```yaml
_ppo_atari_arc: &ppo_atari_arc
  modules:
    body:
      Sequential:
        - LazyConv2d: {out_channels: 32, kernel_size: 8, stride: 4, padding: 0}
        - ReLU:
        - LazyConv2d: {out_channels: 64, kernel_size: 4, stride: 2, padding: 0}
        - ReLU:
        - LazyConv2d: {out_channels: 64, kernel_size: 3, stride: 1, padding: 0}
        - ReLU:
        - Flatten:
        - LazyLinear: {out_features: 512}
        - ReLU:
  graph:
    input: x
    modules:
      body: [x]
    output: body
```

This is the Nature CNN architecture. With TorchArc, the conv layers, flatten, and FC head are all visible in one place.

## Adding Normalization Layers

Insert any `torch.nn` module into the Sequential list. For example, BatchNorm for CrossQ critics:

```yaml
_crossq_critic: &crossq_critic
  modules:
    body:
      Sequential:
        - LazyLinear: {out_features: 2048}
        - LazyBatchNorm1d: {}
        - ReLU: {}
        - LazyLinear: {out_features: 2048}
        - LazyBatchNorm1d: {}
        - ReLU: {}
  graph:
    input: x
    modules:
      body: [x]
    output: body
```

Or LayerNorm:

```yaml
Sequential:
  - LazyLinear: {out_features: 256}
  - LayerNorm: {normalized_shape: 256}
  - ReLU:
  - LazyLinear: {out_features: 256}
  - LayerNorm: {normalized_shape: 256}
  - ReLU:
```

{% hint style="info" %}
With TorchArc, adding normalization is a one-line insertion per layer—no code changes needed.
{% endhint %}

## Old vs. New

| | MLPNet (old) | TorchArcNet (new) |
|---|---|---|
| **Definition** | `hid_layers: [256, 256]` | Explicit YAML modules + graph |
| **Reuse** | Copy-paste across specs | YAML anchors (`&` / `*`) |
| **Flexibility** | Fixed patterns (MLP, Conv, RNN) | Any `torch.nn` module |
| **Transparency** | Layers are implicit | Every layer is visible |
| **Normalization** | `batch_norm: true` flag | Insert `BatchNorm1d`, `LayerNorm`, etc. directly |

{% hint style="success" %}
All `benchmark_arc/` specs use TorchArcNet. The old `MLPNet`/`ConvNet` types still work for backward compatibility, but TorchArc is the recommended approach for new specs.
{% endhint %}

## What's Next

- [Benchmark Specs](../../using-slm-lab/benchmark-specs.md) — see TorchArc specs in action
- [torcharc library](https://github.com/kengz/torcharc) — full documentation for the YAML spec format
- Browse [`slm_lab/spec/benchmark_arc/`](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark_arc) for all benchmark TorchArc specs
