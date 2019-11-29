# Continuous Environment Benchmark

## 🥇 Continuous Environment Benchmark Result

* [Upload PR \#427](https://github.com/kengz/SLM-Lab/pull/427)
* [Dropbox data](https://www.dropbox.com/s/xaxybertpwt4s9j/benchmark_continuous_2019_09.zip?dl=0)

| Env. \ Alg. | A2C \(GAE\) | A2C \(n-step\) | PPO | SAC |
| ---: | :---: | :---: | :---: | :---: |
| RoboschoolAnt | 787 | 1396 | 1843 | **2915** |
| RoboschoolAtlasForwardWalk | 59.87 | 88.04 | 172 | **800** |
| RoboschoolHalfCheetah | 712 | 439 | 1960 | **2497** |
| RoboschoolHopper | 710 | 285 | 2042 | **2045** |
| RoboschoolInvertedDoublePendulum | 996 | 4410 | 8076 | **8085** |
| RoboschoolInvertedPendulum | **995** | 978 | 986 | 941 |
| RoboschoolReacher | 12.9 | 10.16 | 19.51 | **19.99** |
| RoboschoolWalker2d | 280 | 220 | 1660 | **1894** |
| RoboschoolHumanoid | 99.31 | 54.58 | 2388 | **2621\*** |
| RoboschoolHumanoidFlagrun | 73.57 | 178 | 2014 | **2056\*** |
| RoboschoolHumanoidFlagrunHarder | -429 | 253 | **680** | 280\* |
| Unity3DBall | 33.48 | 53.46 | 78.24 | **98.44** |
| Unity3DBallHard | 62.92 | 71.92 | 91.41 | **97.06** |

> Episode score at the end of training attained by SLM Lab implementations on continuous control problems. Reported episode scores are the average over the last 100 checkpoints, and then averaged over 4 Sessions. Results marked with `*` require 50M-100M frames, so we use the hybrid synchronous/asynchronous version of SAC to parallelize and speed up training time.

## 📈 Continuous Environment Benchmark Result Plots

#### Plot Legend

![legend](https://user-images.githubusercontent.com/8209263/67737544-d727dc80-f9c8-11e9-904a-319b9aafd41b.png)

![](https://user-images.githubusercontent.com/8209263/67737923-1571cb80-f9ca-11e9-8f6b-b288fa19bff0.png) ![](https://user-images.githubusercontent.com/8209263/67737924-1571cb80-f9ca-11e9-98ee-82c920dfbf44.png) ![](https://user-images.githubusercontent.com/8209263/67737925-1571cb80-f9ca-11e9-9c7f-3a8294a517af.png) ![](https://user-images.githubusercontent.com/8209263/67737926-160a6200-f9ca-11e9-8cae-9afc532e5af8.png) ![](https://user-images.githubusercontent.com/8209263/67737927-160a6200-f9ca-11e9-8eb2-e04554e3844f.png) ![](https://user-images.githubusercontent.com/8209263/67737928-160a6200-f9ca-11e9-8eae-e7a3ccbe914a.png) ![](https://user-images.githubusercontent.com/8209263/67737929-160a6200-f9ca-11e9-9423-b27165def32e.png) ![](https://user-images.githubusercontent.com/8209263/67737930-160a6200-f9ca-11e9-9a0f-edbd4f01f4e0.png) ![](https://user-images.githubusercontent.com/8209263/67737931-16a2f880-f9ca-11e9-9340-fe90ab48e95f.png) ![](https://user-images.githubusercontent.com/8209263/67737932-16a2f880-f9ca-11e9-92bb-9c896ec3991e.png) ![](https://user-images.githubusercontent.com/8209263/67737933-16a2f880-f9ca-11e9-98c8-7388fa9e1775.png) ![](https://user-images.githubusercontent.com/8209263/67737934-16a2f880-f9ca-11e9-912b-37c8840d0acc.png) ![](https://user-images.githubusercontent.com/8209263/67737935-16a2f880-f9ca-11e9-9275-f3b5fef22e1b.png)

