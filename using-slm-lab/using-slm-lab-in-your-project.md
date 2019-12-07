# Using SLM Lab In Your Project

## SLM Lab As a Python Pip Module

{% hint style="info" %}
This is an advanced use case tutorial.
{% endhint %}

The modular design of SLM Lab implies that its components can also be used in other projects. In fact, SLM Lab can be installed as a pip module and used like a typical Python package. This means you can just do a `pip import slm_lab` \(or`python setup.py install` without `pip`\) and initialize an RL agent for use in your Python project.

This is especially crucial for those who want to use these algorithms in an industrial application. Often the app is part of a massive industrial system, and it is difficult or impossible to wrap that inside the lab. The agent must be made into an importable module to be used inside the app, either for training or for inferencing in deployment.

For demonstration, we have created a standalone script to show how to do this. The solution is very lightweight. The proper spec format will initialize the agent as usual, and as long as the proper agent APIs are called, all agent functionalities will work. Of course, to make use of the lab’s full potential such as distributed training and parameter search, you would still need to use SLM Lab.

The demo below uses a simplified form of the SLM Lab’s Session class. This shows that the main control loop and API methods are already generic.

```bash
# Installation:
# 1. Clone SLM-Lab
git clone https://github.com/kengz/SLM-Lab.git
cd SLM-Lab

# 2. Install SLM Lab as pip module
pip install -e .
```

Let's see how we can implement a Session in an external project.

```python
import os
# NOTE increase if needed. Pytorch thread overusage https://github.com/pytorch/pytorch/issues/975
os.environ['OMP_NUM_THREADS'] = '1'
from slm_lab.spec import spec_util
from slm_lab.lib import logger, util
from slm_lab.experiment import analysis
from slm_lab.env.openai import OpenAIEnv
from slm_lab.agent import Agent, Body
import torch


class Session:
    '''A very simple Session that runs an RL loop'''

    def __init__(self, spec):
        self.spec = spec
        self.env = OpenAIEnv(self.spec)
        body = Body(self.env, self.spec)
        self.agent = Agent(self.spec, body=body)
        logger.info(f'Initialized session')

    def run_rl(self):
        clock = self.env.clock
        state = self.env.reset()
        done = False
        while clock.get('frame') <= self.env.max_frame:
            if done:  # reset when episode is done
                clock.tick('epi')
                state = self.env.reset()
                done = False
            clock.tick('t')
            with torch.no_grad():
                action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            self.agent.update(state, action, reward, next_state, done)
            state = next_state
            if clock.get('frame') % self.env.log_frequency == 0:
                self.agent.body.ckpt(self.env, 'train')
                self.agent.body.log_summary('train')

    def close(self):
        self.agent.close()
        self.env.close()
        logger.info('Session done and closed.')

    def run(self):
        self.run_rl()
        # this will run SLM Lab's built-in analysis module and plot graphs
        self.data = analysis.analyze_session(self.spec, self.agent.body.train_df, 'train')
        self.close()
        return self.data


# This uses SLM-Lab's existing spec. Alternatively, you can write one yourself too (see documentation for detail).
spec = spec_util.get(spec_file='slm_lab/spec/demo.json', spec_name='dqn_cartpole')
os.environ['lab_mode'] = 'train'  # set to 'dev' for rendering

# update the tracking indices
spec_util.tick(spec, 'trial')
spec_util.tick(spec, 'session')

# initialize and run session
session = Session(spec)
session_metrics = session.run()
```

