# Train: REINFORCE CartPole

## 🚀 Train Mode

This tutorial will look at how to train an agent in SLM Lab, then use the automatically-saved model files to replay it in enjoy mode, using REINFORCE on CartPole.

REINFORCE is a very basic policy gradient algorithm, and we are training it on an easy environment. We will use the spec file at [slm\_lab/spec/benchmark/reinforce/reinforce\_cartpole.json](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json), in particular the **reinforce\_cartpole** spec, and run it in **train** mode. The lab command is:

```bash
python run_lab.py slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json reinforce_cartpole train
```

This will run a `Trial` with 4 `Sessions` of different random seeds to average the results. Wait for it to run until completion, which should take about 10-20 minutes. Meanwhile, check the metrics logged in the terminal. In particular, the `total_reward` and its moving average `total_reward_ma` \(with a window of 100 episodes\) should climb up gradually.

{% hint style="success" %}
REINFORCE successfully trains on CartPole when the `total_reward_ma` reaches close to the maximum of 200, although a score of over 100 will do for this tutorial.
{% endhint %}

When the trial completes, all the metrics, graphs and data will be saved to a timestamped folder, let's say `data/reinforce_cartpole_2020_04_13_232521/`. Among other things, SLM Lab also automatically saves the **final** and the **best** model files in the model folder. The model files can be used for easy playback in enjoy mode.

Next, we look at how to resume training, and to run a trained model in enjoy mode.

