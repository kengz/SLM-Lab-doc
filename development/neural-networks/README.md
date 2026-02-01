# Net 🕸️

## Overview

Net classes implement neural network architectures used as function approximators in RL algorithms. SLM Lab provides flexible, swappable networks that work with any algorithm.

**Code:** [slm\_lab/agent/net](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/net)

## Network Types

| Type | Input | Use Case | Example Environments |
|------|-------|----------|----------------------|
| [**MLPNet**](mlp.md) | Vectors | Low-dimensional states | CartPole, LunarLander, MuJoCo |
| [**ConvNet**](cnn.md) | Images | Pixel observations | Atari games |
| [**RecurrentNet**](rnn.md) | Sequences | Partial observability | POMDPs |
| **HydraMLPNet** | Multiple vectors | Multi-head architectures | Multi-task learning |
| **DuelingMLPNet** | Vectors | Q-learning | LunarLander (value decomposition) |
| **DuelingConvNet** | Images | Q-learning | Atari (value decomposition) |

## Quick Selection Guide

```
Is your observation an image?
├── Yes → ConvNet
└── No → Is there partial observability?
    ├── Yes → RecurrentNet
    └── No → MLPNet
```

For Q-learning algorithms (DQN family), consider Dueling variants for better value estimation.

## Network Spec

Configure networks in the agent spec:

```javascript
{
  "agent": {
    "net": {
      // Network type
      "type": "MLPNet",

      // Architecture
      "hid_layers": [256, 256],           // Hidden layer sizes
      "hid_layers_activation": "relu",    // Activation function

      // Training
      "optim_spec": {                     // Optimizer
        "name": "Adam",
        "lr": 3e-4
      },
      "clip_grad_val": 0.5,               // Gradient clipping

      // Device
      "gpu": "auto"                       // "auto", true, false
    }
  }
}
```

## Common Parameters

### Architecture

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `type` | Network class | `"MLPNet"`, `"ConvNet"`, `"RecurrentNet"` |
| `hid_layers` | Hidden layer sizes | `[64, 64]` (simple), `[256, 256]` (complex) |
| `hid_layers_activation` | Activation function | `"relu"`, `"tanh"`, `"leaky_relu"` |
| `out_layer_activation` | Output activation | `null` (none), `"tanh"` |
| `init_fn` | Weight initialization | `"orthogonal_"`, `"xavier_uniform_"` |

### Actor-Critic Networks

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `shared` | Share weights between actor/critic | `true` (Atari), `false` (MuJoCo) |
| `use_same_optim` | Use same optimizer for both | `true`, `false` |
| `actor_optim_spec` | Actor optimizer | `{"name": "Adam", "lr": 3e-4}` |
| `critic_optim_spec` | Critic optimizer | `{"name": "Adam", "lr": 3e-4}` |

### Training

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `clip_grad_val` | Gradient clipping norm | 0.5-10.0 |
| `loss_spec` | Loss function | `{"name": "MSELoss"}`, `{"name": "SmoothL1Loss"}` |
| `lr_scheduler_spec` | Learning rate schedule | See below |

### ConvNet Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `normalize` | Normalize pixel input by dividing by 255 | `false` |
| `batch_norm` | Apply batch normalization after conv layers | `true` (ConvNet), `false` (DuelingConvNet) |

### Device

| Parameter | Description | Values |
|-----------|-------------|--------|
| `gpu` | GPU usage | `"auto"` (detect), `true` (force), `false` (CPU only) |

## Learning Rate Schedules

Decay learning rate during training:

```javascript
{
  "lr_scheduler_spec": {
    "name": "LinearToZero",  // Linear decay to 0
    "frame": 1000000         // Total frames for decay
  }
}
```

Available schedules:
- `"LinearToZero"` - Linear decay from initial LR to 0
- `"StepLR"` - Step decay at fixed intervals
- `"ExponentialLR"` - Exponential decay

## Example Specs

### MLP for CartPole/MuJoCo

```javascript
{
  "net": {
    "type": "MLPNet",
    "shared": false,
    "hid_layers": [64, 64],
    "hid_layers_activation": "tanh",
    "init_fn": "orthogonal_",
    "clip_grad_val": 0.5,
    "loss_spec": {"name": "MSELoss"},
    "actor_optim_spec": {"name": "Adam", "lr": 3e-4},
    "critic_optim_spec": {"name": "Adam", "lr": 3e-4},
    "gpu": "auto"
  }
}
```

### ConvNet for Atari (Nature CNN)

```javascript
{
  "net": {
    "type": "ConvNet",
    "shared": true,
    "conv_hid_layers": [
      [32, 8, 4, 0, 1],  // [out_channels, kernel, stride, padding, dilation]
      [64, 4, 2, 0, 1],
      [64, 3, 1, 0, 1]
    ],
    "fc_hid_layers": [512],
    "hid_layers_activation": "relu",
    "init_fn": "orthogonal_",
    "normalize": true,
    "clip_grad_val": 0.5,
    "use_same_optim": true,
    "optim_spec": {"name": "AdamW", "lr": 2.5e-4},
    "lr_scheduler_spec": {"name": "LinearToZero", "frame": 10e6},
    "gpu": "auto"
  }
}
```

### RecurrentNet for POMDPs

```javascript
{
  "net": {
    "type": "RecurrentNet",
    "cell_type": "GRU",
    "fc_hid_layers": [128],
    "rnn_hidden_size": 64,
    "rnn_num_layers": 1,
    "seq_len": 8,
    "hid_layers_activation": "relu",
    "optim_spec": {"name": "Adam", "lr": 1e-3},
    "gpu": "auto"
  }
}
```

### DQN Target Network

For DQN algorithms, a separate target network is created automatically:

```javascript
{
  "net": {
    "type": "MLPNet",
    "hid_layers": [256, 128],
    "update_type": "replace",     // How to update target
    "update_frequency": 100,      // Steps between updates
    // Or use soft updates:
    // "update_type": "polyak",
    // "polyak_coef": 0.995
  }
}
```

## Network Architecture Tips

### CartPole / Simple Control

```javascript
{"hid_layers": [64, 64], "hid_layers_activation": "tanh"}
```
- Small networks work well
- `tanh` activation is common for bounded outputs

### MuJoCo / Continuous Control

```javascript
{"hid_layers": [256, 256], "hid_layers_activation": "tanh", "init_fn": "orthogonal_"}
```
- Larger networks for complex dynamics
- Orthogonal initialization helps with gradients

### Atari / Image-Based

```javascript
{
  "conv_hid_layers": [[32, 8, 4, 0, 1], [64, 4, 2, 0, 1], [64, 3, 1, 0, 1]],
  "fc_hid_layers": [512]
}
```
- Nature CNN architecture is standard
- `shared: true` for actor-critic

### Box2D (LunarLander, BipedalWalker)

```javascript
{"hid_layers": [256, 128], "hid_layers_activation": "relu"}
```
- Medium-sized networks
- `relu` works well

## GPU Usage

SLM Lab handles GPU placement automatically:

```javascript
{"gpu": "auto"}  // Use GPU if available, else CPU
{"gpu": true}    // Force GPU (fails if unavailable)
{"gpu": false}   // Force CPU
{"gpu": 0}       // Specific GPU device
```

For multi-GPU setups, see [GPU Usage: PPO on Pong](../../using-slm-lab/gpu-usage-ppo-on-pong.md).
