# Using SLM Lab In Your Project 🔧

{% hint style="info" %}
This is an advanced tutorial for integrating SLM Lab agents into your own applications.
{% endhint %}

SLM Lab's modular design allows its components to be used outside the standard lab framework. This is useful when:

- Building an RL-powered application that integrates with existing systems
- Creating custom training loops with domain-specific logic
- Using SLM Lab's well-tested algorithms in a different framework

## Installation

```bash
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab
uv sync
```

## Key Components

| Component | Import | Purpose |
|-----------|--------|---------|
| `spec_util` | `from slm_lab.spec import spec_util` | Load specs from files |
| `util` | `from slm_lab.lib import util` | Utilities including `read()` for saved specs |
| `make_env` | `from slm_lab.env import make_env` | Create environments with SLM Lab wrappers |
| `Agent` | `from slm_lab.agent import Agent` | The RL agent |
| `MetricsTracker` | `from slm_lab.agent import MetricsTracker` | Metrics collection and logging |
| `net_util` | `from slm_lab.agent.net import net_util` | Network utilities including model loading |

## Example: Custom Training Loop

Here's a complete example of running a training loop outside the lab:

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent PyTorch thread overuse

import numpy as np
import torch

from slm_lab.spec import spec_util
from slm_lab.lib import logger, util
from slm_lab.experiment import analysis
from slm_lab.env import make_env
from slm_lab.agent import Agent, MetricsTracker


class CustomSession:
    """A simplified Session that runs an RL training loop."""

    def __init__(self, spec):
        self.spec = spec
        self.env = make_env(self.spec)
        self.mt = MetricsTracker(self.env, self.spec)
        self.agent = Agent(self.spec, mt=self.mt)
        logger.info(f'Initialized session for {spec["name"]}')

    def run_rl(self):
        """Main RL training loop."""
        state, info = self.env.reset()

        while self.env.get() < self.env.max_frame:
            with torch.no_grad():
                action = self.agent.act(state)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = np.logical_or(terminated, truncated)

            self.agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                terminated=terminated,
                truncated=truncated
            )

            # Periodic logging
            if util.frame_mod(self.env.get(), self.env.log_frequency, self.env.num_envs):
                self.mt.ckpt(self.env, 'train')
                self.mt.log_summary('train')

            # Handle episode reset (only for single env; vector envs auto-reset)
            if util.epi_done(done):
                state, info = self.env.reset()
            else:
                state = next_state

    def close(self):
        self.agent.close()
        self.env.close()
        logger.info('Session done.')

    def run(self):
        self.run_rl()
        self.data = analysis.analyze_session(self.spec, self.mt.train_df, 'train')
        self.close()
        return self.data


if __name__ == '__main__':
    # Load a spec
    spec = spec_util.get(
        spec_file='slm_lab/spec/benchmark/ppo/ppo_cartpole.json',
        spec_name='ppo_cartpole'
    )

    # Set lab mode
    os.environ['lab_mode'] = 'train'

    # Initialize indices (required for file naming)
    spec_util.tick(spec, 'trial')
    spec_util.tick(spec, 'session')

    # Run
    session = CustomSession(spec)
    metrics = session.run()
    print(f"Final metrics: {metrics}")
```

## Example: Inference with Trained Agent

To use a trained agent for inference:

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['lab_mode'] = 'enjoy'  # Inference mode

import numpy as np
import torch

from slm_lab.lib import util
from slm_lab.spec import spec_util
from slm_lab.env import make_env
from slm_lab.agent import Agent, MetricsTracker
from slm_lab.agent.net import net_util


def load_trained_agent(spec_path: str):
    """Load a trained agent from a saved trial spec."""
    # Load saved spec (includes model paths)
    spec = util.read(spec_path)

    # Initialize indices
    spec_util.tick(spec, 'trial')
    spec_util.tick(spec, 'session')

    # Create env and agent
    env = make_env(spec)
    mt = MetricsTracker(env, spec)
    agent = Agent(spec, mt=mt)

    # Load trained weights (uses model_prepath from spec)
    net_util.load_algorithm(agent.algorithm)

    return agent, env


def run_inference(agent, env, num_episodes: int = 10):
    """Run the agent and collect rewards."""
    total_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = agent.act(state)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: reward = {episode_reward}")

    print(f"Average: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    return total_rewards


if __name__ == '__main__':
    # Use the trial spec (contains model paths)
    agent, env = load_trained_agent(
        'data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_spec.json'
    )
    rewards = run_inference(agent, env, num_episodes=10)
    env.close()
```

## Key APIs

### Agent

```python
agent.act(state)              # Returns action given state
agent.update(state, action, reward, next_state, done, terminated, truncated)
agent.save(ckpt='best')       # Save checkpoint ('best' or None for regular)
agent.close()                 # Cleanup and final save
```

### Environment

```python
env.reset()                   # Returns (state, info)
env.step(action)              # Returns (state, reward, terminated, truncated, info)
env.get()                     # Current frame count
env.max_frame                 # Total training frames
env.log_frequency             # Logging interval
env.close()                   # Cleanup
```

### Spec Utilities

```python
spec_util.get(spec_file, spec_name)       # Load spec from file
spec_util.get(..., sets=['env=Hopper-v5']) # With variable substitution
spec_util.tick(spec, 'trial')             # Increment trial/session index
util.read(spec_path)                      # Load saved spec from experiment
```

### Model Loading

```python
net_util.load_algorithm(agent.algorithm)  # Load all nets for algorithm
net_util.load(net, model_path)            # Load single net from path
```

## Tips

1. **Thread management**: Set `OMP_NUM_THREADS=1` to prevent PyTorch thread overuse
2. **Lab mode**: Set `os.environ['lab_mode']` before creating agents:
   - `'train'`: Full training with checkpoints
   - `'dev'`: Training with rendering
   - `'enjoy'`: Inference only (loads best checkpoint)
3. **GPU usage**: The `gpu` field in net spec controls device:
   - `"auto"`: Use GPU if available
   - `true`: Force GPU
   - `false`: CPU only

## Limitations

When using SLM Lab as a module, you lose:

- Ray Tune hyperparameter search
- Automatic experiment organization
- TensorBoard integration

For these features, use the full framework via `slm-lab run`.
