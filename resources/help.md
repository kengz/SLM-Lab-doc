# Help

## NVIDIA GPU driver problem

If you receive errors similar to the following when trying to use GPU:

> NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver

Reinstall your NVIDIA GPU driver using [this instruction](https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07).

## Building and setting up a Linux GPU server

If you build your own desktop and want a quick and smooth setup for a Ubuntu GPU server, refer to [this gist](https://gist.github.com/kengz/a106e03a782cfaec339433daf8965d76).

## Breakage from SLM-Lab update

Make sure you also install the packages after updating the repo. Run:

```bash
git pull
uv sync
```

## Search is running slow

In certain setup, the search mode's parallel processing may run slower because of race condition in PyTorch's greedy CPU utilization. This is indicated when the logged fps (frame-per-second) is much slower in search than when simply training a trial, e.g. fps 200 vs 10.

This issue is documented here:

* [https://github.com/pytorch/pytorch/issues/3146](https://github.com/pytorch/pytorch/issues/3146)
* [https://github.com/HumanCompatibleAI/imitation/issues/274](https://github.com/HumanCompatibleAI/imitation/issues/274)

To fix it, prepend an `OMP_NUM_THREADS=1` to the run command. For example:

```bash
OMP_NUM_THREADS=1 slm-lab run slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole search
```

## How to kill stuck processes?

You can see the running processes using tools like [glances](https://github.com/nicolargo/glances). Use the following commands to kill processes by their names. You may need to use `sudo`.

```bash
pkill -f slm-lab
pkill -f ray
pkill -f Xvfb
```

Or use the built-in command:

```bash
slm-lab run --stop-ray
```

## No GUI or images saved on a headless remote server

When running SLM Lab on a remote server, you may get `NoSuchDisplayException: Cannot connect to "None"`. Or your graphs may not be generated. This is because servers are typically headless, i.e. without a display. This error occurs when you're trying to render without a headless display.

First, try setting environment variable `RENDER=false` before the lab command, for example:

```bash
RENDER=false slm-lab run slm_lab/spec/demo.json ppo_cartpole train
```

Despite its simplicity, this option comes with the caveat that plots from Plotly cannot generated. The safer option is to install **Xvfb**, and prepend your command with `xvfb-run -a`. For example:

```bash
xvfb-run -a slm-lab run slm_lab/spec/demo.json ppo_cartpole train
```

## How to forward GUI from a remote server?

If you are running via `ssh` and want GUI forwarding from a server, do:

* [install X11 on your server](https://help.ubuntu.com/community/ServerGUI)
* install OpenGL and/or configure Nvidia driver on your server. [Follow instructions here.](https://github.com/openai/gym/issues/468)
* [install XQuartz/Xming on your laptop](https://uisapp2.iu.edu/confluence-prd/pages/viewpage.action?pageId=280461906)
* do `ssh` with a `-X` flag, e.g. `ssh -X foo@bar`.

## How to sync data from a remote server?

SLM Lab produces a lot of data which are then zipped for our convenience of transferring/syncing them. The recommended method is to use HuggingFace for experiment storage. See the [Remote Training](../using-slm-lab/remote-training.md) guide for setup.

```bash
# Push local results to HuggingFace
slm-lab push data/ppo_lunar_2024_01_15_123456

# Pull results from HuggingFace
slm-lab pull ppo_lunar
```

## What is SLM?

SLM stands for _Strange Loop Machine_, in homage to Hofstadter's iconic book [_Gödel, Escher, Bach: An Eternal Golden Braid_](https://www.amazon.com/G%C3%B6del-Escher-Bach-Eternal-Golden/dp/0465026567). This lab is created as part of a long term project to try out AI ideas heavily influenced by it.

## Reporting Issues

Can't find the issues you encountered? [Report new issues on Github](https://github.com/kengz/SLM-Lab/issues); it helps all of us.
