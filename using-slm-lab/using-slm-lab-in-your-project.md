# Using SLM Lab In Your Project 🔧

## SLM Lab As a Python Module

{% hint style="info" %}
This is an advanced use case tutorial for integrating SLM Lab agents into your own applications.
{% endhint %}

The modular design of SLM Lab allows its components to be used outside the standard lab framework. This is useful when:

- Building an RL-powered application that needs to integrate with existing systems
- Creating custom training loops with domain-specific logic
- Using SLM Lab's well-tested algorithms in a different framework

## Installation

```bash
# Clone and install SLM Lab
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab
uv sync
```

## Basic Integration

The key SLM Lab components you'll use:

| Component | Import | Purpose |
|-----------|--------|---------|
| `spec_util` | `from slm_lab.spec import spec_util` | Load and manage spec files |
| `make_env` | `from slm_lab.env import make_env` | Create Gymnasium environments with SLM Lab wrappers |
| `Agent` | `from slm_lab.agent import Agent` | The trained agent |
| `MetricsTracker` | `from slm_lab.agent import MetricsTracker` | Metrics collection |

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
        # Create environment with SLM Lab wrappers
        self.env = make_env(self.spec)
        # Create metrics tracker (handles logging, checkpoints)
        self.mt = MetricsTracker(self.env, self.spec)
        # Create agent (contains algorithm, memory, networks)
        self.agent = Agent(self.spec, mt=self.mt)
        logger.info(f'Initialized session for {spec["name"]}')

    def run_rl(self):
        """Main RL training loop."""
        state, info = self.env.reset()

        while self.env.get() < self.env.max_frame:
            # Select action
            with torch.no_grad():
                action = self.agent.act(state)

            # Execute in environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = np.logical_or(terminated, truncated)

            # Update agent (stores experience, trains if ready)
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

            # Handle episode reset
            if util.epi_done(done):
                state, info = self.env.reset()
            else:
                state = next_state

    def close(self):
        """Cleanup resources."""
        self.agent.close()
        self.env.close()
        logger.info('Session done and closed.')

    def run(self):
        """Run training and return metrics."""
        self.run_rl()
        # Analyze results using SLM Lab's analysis module
        self.data = analysis.analyze_session(self.spec, self.mt.train_df, 'train')
        self.close()
        return self.data


# Usage example
if __name__ == '__main__':
    # Load a spec file
    spec = spec_util.get(
        spec_file='slm_lab/spec/demo.json',
        spec_name='ppo_cartpole'
    )

    # Set lab mode: 'train' for training, 'dev' for rendering
    os.environ['lab_mode'] = 'train'

    # Initialize tracking indices (required for file naming)
    spec_util.tick(spec, 'trial')
    spec_util.tick(spec, 'session')

    # Run training
    session = CustomSession(spec)
    metrics = session.run()

    print(f"Final metrics: {metrics}")
```

## Example: Using a Trained Agent for Inference

To use a trained agent for inference (e.g., in a deployed application):

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch

from slm_lab.spec import spec_util
from slm_lab.env import make_env
from slm_lab.agent import Agent, MetricsTracker
from slm_lab.agent.net import net_util


def load_trained_agent(spec_path: str, model_path: str):
    """Load a trained agent from saved spec and model files."""
    # Load the spec
    spec = spec_util.get_from_file(spec_path)

    # Set eval mode (no training, no model saving)
    os.environ['lab_mode'] = 'eval'

    # Initialize indices
    spec_util.tick(spec, 'trial')
    spec_util.tick(spec, 'session')

    # Create env and agent
    env = make_env(spec)
    mt = MetricsTracker(env, spec)
    agent = Agent(spec, mt=mt)

    # Load trained weights
    net_util.load(agent.algorithm, model_path)

    return agent, env


def run_inference(agent, env, num_episodes: int = 10):
    """Run the agent for inference and collect rewards."""
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

    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    return total_rewards


# Usage
if __name__ == '__main__':
    agent, env = load_trained_agent(
        spec_path='data/ppo_cartpole_2026_01_30_221924/ppo_cartpole_t0_spec.json',
        model_path='data/ppo_cartpole_2026_01_30_221924/model/ppo_cartpole_t0_s0_ckpt-best_net_model.pt'
    )
    rewards = run_inference(agent, env, num_episodes=10)
    env.close()
```

## Key APIs

### Agent API

```python
agent.act(state)          # Returns action given state
agent.update(...)         # Update memory and train
agent.save(ckpt='best')   # Save model checkpoint
agent.close()             # Cleanup and final save
```

### Environment API

```python
env.reset()               # Returns (state, info)
env.step(action)          # Returns (state, reward, terminated, truncated, info)
env.get()                 # Get current frame count
env.close()               # Cleanup
```

### Spec Utilities

```python
spec_util.get(spec_file, spec_name)  # Load spec from file
spec_util.get_from_file(spec_path)   # Load spec from saved experiment
spec_util.tick(spec, 'trial')        # Increment trial/session index
```

## Tips

1. **Thread management**: Set `OMP_NUM_THREADS=1` to prevent PyTorch from over-allocating threads
2. **Lab mode**: Set `os.environ['lab_mode']` to control behavior:
   - `'train'`: Full training with checkpoints
   - `'dev'`: Training with rendering
   - `'eval'`: Inference only, no saving
3. **GPU usage**: The `gpu` field in net spec controls device placement:
   - `"auto"`: Use GPU if available
   - `true`: Force GPU (fails if unavailable)
   - `false`: CPU only

## Limitations

When using SLM Lab as a module, you lose access to:

- Distributed training (Hogwild!)
- Ray Tune hyperparameter search
- Automatic experiment organization

For these features, use the full SLM Lab framework via `slm-lab run`.
