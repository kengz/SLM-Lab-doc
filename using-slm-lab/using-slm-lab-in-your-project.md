# Using SLM Lab In Your Project

## SLM Lab As a Python Module

{% hint style="info" %}
This is an advanced use case tutorial.
{% endhint %}

The modular design of SLM Lab implies that its components can also be used in other projects. SLM Lab can be installed as a Python package and used in your own project. This means you can import `slm_lab` and initialize an RL agent for training or inference.

This is especially useful for those who want to use these algorithms in an industrial application. Often the app is part of a larger system, and it is difficult or impossible to wrap that inside the lab. The agent must be made into an importable module to be used inside the app.

For demonstration, we have created a standalone script to show how to do this. The solution is very lightweight. The proper spec format will initialize the agent as usual, and as long as the proper agent APIs are called, all agent functionalities will work. Of course, to make use of the lab's full potential such as distributed training and parameter search, you would still need to use SLM Lab.

The demo below uses a simplified form of the SLM Lab's Session class. This shows that the main control loop and API methods are already generic.

```bash
# Installation:
# 1. Clone SLM-Lab
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab

# 2. Install SLM Lab
uv sync
```

Let's see how we can implement a Session in an external project.

```python
import os
# NOTE increase if needed. Pytorch thread overusage https://github.com/pytorch/pytorch/issues/975
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch

from slm_lab.spec import spec_util
from slm_lab.lib import logger, util
from slm_lab.experiment import analysis
from slm_lab.env import make_env
from slm_lab.agent import Agent, MetricsTracker


class Session:
    '''A simple Session that runs an RL loop'''

    def __init__(self, spec):
        self.spec = spec
        self.env = make_env(self.spec)
        self.mt = MetricsTracker(self.env, self.spec)
        self.agent = Agent(self.spec, mt=self.mt)
        logger.info(f'Initialized session')

    def run_rl(self):
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

            # Checkpoint and log
            if util.frame_mod(self.env.get(), self.env.log_frequency, self.env.num_envs):
                self.mt.ckpt(self.env, 'train')
                self.mt.log_summary('train')

            if util.epi_done(done):
                state, info = self.env.reset()
            else:
                state = next_state

    def close(self):
        self.agent.close()
        self.env.close()
        logger.info('Session done and closed.')

    def run(self):
        self.run_rl()
        # Run SLM Lab's built-in analysis module
        self.data = analysis.analyze_session(self.spec, self.mt.train_df, 'train')
        self.close()
        return self.data


# This uses SLM-Lab's existing spec
spec = spec_util.get(spec_file='slm_lab/spec/demo.json', spec_name='ppo_cartpole')
os.environ['lab_mode'] = 'train'  # set to 'dev' for rendering

# Update tracking indices
spec_util.tick(spec, 'trial')
spec_util.tick(spec, 'session')

# Initialize and run session
session = Session(spec)
session_metrics = session.run()
```
