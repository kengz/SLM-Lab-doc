# Using SLM Lab In Your Project

## SLM Lab As a Python Pip Module

{% hint style="info" %}
This is an advanced use case tutorial.
{% endhint %}

The modular design of SLM Lab implies that its components can also be used in other projects. In fact, SLM Lab can be installed as a pip module and used like a typical Python package. This means you can just do a `pip import slm_lab` \(or`python setup.py install` without `pip`\) and initialize an RL agent for use in your Python project.

This is especially crucial for those who want to use these algorithms in an industrial application. Often the app is part of a massive industrial system, and it is difficult or impossible to wrap that inside the lab. The agent must be made into an importable module to be used inside the app, either for training or for inferencing in deployment.

For demonstration, we have created a standalone script to show how to do this. The solution is very lightweight. The proper spec format will initialize the agent as usual, and as long as the proper agent APIs are called, all agent functionalities will work. Of course, to make use of the lab’s full potential such as distributed training, parameter search, data analysis, you would still need to use SLM Lab.

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
from slm_lab.agent import Agent
from slm_lab.env import OpenAIEnv
from slm_lab.experiment import analysis
from slm_lab.experiment.monitor import Body
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util


class Session:
    '''A very simple Session that runs an RL loop'''

    def __init__(self, spec):
        self.spec = spec
        self.env = OpenAIEnv(self.spec)
        body = Body(self.env, self.spec['agent'])
        self.agent = Agent(self.spec, body=body)
        logger.info(f'Initialized session')

    def run_episode(self):
        self.env.clock.tick('epi')
        reward, state, done = self.env.reset()
        self.agent.reset(state)
        while not done:
            self.env.clock.tick('t')
            action = self.agent.act(state)
            reward, state, done = self.env.step(action)
            self.agent.update(action, reward, state, done)
        self.agent.body.log_summary()

    def close(self):
        self.agent.close()
        self.env.close()
        logger.info('Session done and closed.')

    def run(self):
        while self.env.clock.get('epi') <= self.env.max_episode:
            self.run_episode()
        self.data = analysis.analyze_session(self)  # session fitness
        self.close()
        return self.data


# To use SLM-Lab's existing spec. Alternatively, you can write one yourself too
spec = spec_util.get(spec_file='slm_lab/spec/benchmark/ppo/ppo_cartpole.json', spec_name='ppo_shared_cartpole')
os.environ['lab_mode'] = 'train'  # set to 'dev' for rendering

# inspect the agent spec; edit if you wish to
print(spec['agent'])
# edit the env spec to run for less episodes
spec['env'][0]['max_episode'] = 100

# initialize and run session
sess = Session(spec)
data = sess.run()
```

