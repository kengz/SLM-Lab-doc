# Public Benchmark Data

## 📂 Public Data

SLM Lab provides a set of benchmark results that are periodically updated with new feature releases. All the result data is [uploaded from a Pull Request](https://github.com/kengz/SLM-Lab/pulls?utf8=%E2%9C%93&q=is%3Apr+label%3Aresult+) and made [public on Dropbox](https://www.dropbox.com/sh/urifraklxcvol70/AADxtt6zUNuVR6qe288JYNCNa?dl=0).

The data can be downloaded and unzipped into SLM Lab's `data/` folder and rerun in [enjoy mode](../using-slm-lab/train-and-enjoy-dqn-cartpole.md).

## 📌 Benchmark Information

### **Hardware**

For reference, the image based environment benchmarks are run on AWS GPU box `p2.16xlarge`, and the non-image based environments are run on AWS CPU box `m5a.24xlarge`.

### **Reproducibility**

The benchmark tables in this page show the `Trial` level `final_return_ma` from SLM Lab. This is final value of the 100-ckpt moving average of the return \(total rewards\) from evaluation. Each `Trial` is ran with 4 `Session`s with different random seeds, and their `final_return_ma` are averaged on the `Trial` level.

The specs for these are contained in the [`slm_lab/spec/benchmark`](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark) folder, descriptively named `{algorithm}_{environment}.json`. They can be exactly reproduced as described in [Lab Organization](../using-slm-lab/lab-organization.md#reproducibility-design).

### **Environments**

SLM Lab's benchmark includes environments from the following offerings:

* [OpenAI gym default environments](https://github.com/openai/gym)
* [OpenAI gym Atari environments](https://gym.openai.com/envs/#atari) offers a wrapper for the [Atari Learning Environment \(ALE\)](https://github.com/mgbellemare/Arcade-Learning-Environment)
* [OpenAI Roboschool](https://github.com/openai/roboschool)
* [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents)

### **Terminology**

Deep RL algorithms use a lot of abbreviations. Here's a list to help us navigate:

* A2C \(GAE\): Advantage Actor-Critic with GAE as advantage estimation
* A2C \(n-step\): Advantage Actor-Critic with n-step return as advantage estimation
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

