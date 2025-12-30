# Public Benchmark Data

## :open\_file\_folder: Public Data

SLM Lab provides benchmark results that are periodically updated with new releases.

### v5 Results (Current)

New v5 benchmark data is stored on HuggingFace and can be downloaded using the CLI:

```bash
# List available experiments
slm-lab list

# Download specific experiment
slm-lab pull ppo_hopper
slm-lab pull sac_halfcheetah
```

Results are downloaded to the `data/` folder for local analysis or replay in [enjoy mode](../using-slm-lab/train-and-enjoy-dqn-cartpole.md).

{% hint style="info" %}
To use HuggingFace commands, configure `HF_TOKEN` and `HF_REPO` in your `.env` file. See [Remote Training](../using-slm-lab/remote-training.md) for setup.
{% endhint %}

### v4 Results (Historical)

Historical v4 benchmark data remains available on [Google Drive](https://drive.google.com/drive/folders/1fUB3jRvXr8ySZMSW5w0GPWJe3QmM7tb3?usp=sharing). Download and unzip into the `data/` folder to replay.

## :pushpin: Benchmark Information

### **Hardware**

**v5 benchmarks** are run on cloud GPUs via [dstack](https://dstack.ai/) (typically L4 or A10G GPUs).

**v4 historical benchmarks** were run on AWS GPU box `p2.16xlarge` (image-based) and AWS CPU box `m5a.24xlarge` (non-image-based).

### **Reproducibility**

The benchmark tables in this page show the `Trial` level `final_return_ma` from SLM Lab. This is final value of the 100-ckpt moving average of the return (total rewards) from evaluation. Each `Trial` is ran with 4 `Session`s with different random seeds, and their `final_return_ma` are averaged on the `Trial` level.

The specs for these are contained in the [`slm_lab/spec/benchmark`](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark) folder, descriptively named `{algorithm}_{environment}.json`. They can be exactly reproduced as described in [Lab Organization](../using-slm-lab/lab-organization.md#reproducibility-design).

### **Environments**

SLM Lab supports environments from [Gymnasium](https://gymnasium.farama.org/) (the maintained fork of OpenAI Gym):

* **Classic control:** CartPole, Pendulum, Acrobot, MountainCar
* **Box2D:** LunarLander, BipedalWalker
* **MuJoCo:** Hopper, HalfCheetah, Walker2d, Ant, Humanoid, and more
* **Atari:** All 57 games via the [Arcade Learning Environment (ALE)](https://github.com/Farama-Foundation/Arcade-Learning-Environment)

Any gymnasium-compatible environment can be used by specifying its name in the spec file.

### **Terminology**

Deep RL algorithms use a lot of abbreviations. Here's a list to help us navigate:

* A2C (GAE): Advantage Actor-Critic with GAE as advantage estimation
* A2C (n-step): Advantage Actor-Critic with n-step return as advantage estimation
* A3C: Asynchronous Advantage Actor-Critic
* CER: Combined Experience Replay
* DDQN: Double Deep Q-Network
* Async: Asynchronous
* DQN: Deep Q-Network
* GAE: Generalized Advantage Estimation
* PER: Prioritized Experience Replay
* PPO: Proximal Policy Optimization
* SAC: Soft Actor-Critic
* SIL: Self Imitation Learning

Read on to see the benchmark result tables and plots.
