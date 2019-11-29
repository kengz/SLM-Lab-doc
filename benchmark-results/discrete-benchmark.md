# Discrete Environment Benchmark

## 🥇 Discrete Environment Benchmark Result

* [Upload PR \#427](https://github.com/kengz/SLM-Lab/pull/427)
* [Dropbox data](https://www.dropbox.com/s/az4vncwwktyotol/benchmark_discrete_2019_09.zip?dl=0)

| Env. \ Alg. | DQN | DDQN+PER | A2C \(GAE\) | A2C \(n-step\) | PPO | SAC |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Breakout | 80.88 | 182 | 377 | 398 | **443** | 3.51\* |
| Pong | 18.48 | 20.5 | 19.31 | 19.56 | **20.58** | 19.87\* |
| Qbert | 5494 | 11426 | 12405 | **13590** | 13460 | 923\* |
| Seaquest | 1185 | **4405** | 1070 | 1684 | 1715 | 171\* |
| LunarLander | 192 | 233 | 25.21 | 68.23 | 214 | **276** |
| UnityHallway | -0.32 | 0.27 | 0.08 | -0.96 | **0.73** | 0.01 |
| UnityPushBlock | 4.88 | 4.93 | 4.68 | 4.93 | **4.97** | -0.70 |

> Episode score at the end of training attained by SLM Lab implementations on discrete-action control problems. Reported episode scores are the average over the last 100 checkpoints, and then averaged over 4 Sessions. A Random baseline with score averaged over 100 episodes is included. Results marked with `*` were trained using the hybrid synchronous/asynchronous version of SAC to parallelize and speed up training time. For SAC, Breakout, Pong and Seaquest were trained for 2M frames instead of 10M frames.
>
> For the full Atari benchmark, see [Atari Benchmark](https://github.com/kengz/SLM-Lab/blob/benchmark/BENCHMARK.md#atari-benchmark)

## 📈 Discrete Environment Benchmark Result Plots

#### Plot Legend

![legend](https://user-images.githubusercontent.com/8209263/67737544-d727dc80-f9c8-11e9-904a-319b9aafd41b.png)

![](https://user-images.githubusercontent.com/8209263/68744935-84dee200-05aa-11ea-9086-12546f5aa606.png) ![](https://user-images.githubusercontent.com/8209263/68744936-85777880-05aa-11ea-8a6a-4c364a27ba81.png) ![](https://user-images.githubusercontent.com/8209263/68744937-85777880-05aa-11ea-9927-e300309d1e9c.png) ![](https://user-images.githubusercontent.com/8209263/68744939-85777880-05aa-11ea-825c-b1225a0539af.png) ![](https://user-images.githubusercontent.com/8209263/67737566-e7d85280-f9c8-11e9-8df8-39c1205c5308.png) ![](https://user-images.githubusercontent.com/8209263/68744968-90caa400-05aa-11ea-9247-d4d81533965a.png) ![](https://user-images.githubusercontent.com/8209263/68744969-90caa400-05aa-11ea-9b17-45fb852e2671.png)

