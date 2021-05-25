# Help

## Permission denied when running `bin/setup`

This means you don't have sufficient privilege on your machine. Run it with sudo:

```bash
sudo ./bin/setup
```

## `conda activate lab` fails

When Conda complains about certain variables should not be in your `PATH`:

> CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'. If your shell is Bash or a Bourne variant, enable conda for the current user with
>
> $ echo ". /home/ubuntu/miniconda3/etc/profile.d/conda.sh" &gt;&gt; ~/.bashrc
>
> or, for all users, enable conda with
>
> $ sudo ln -s /home/ubuntu/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh
>
> The options above will permanently enable the 'conda' command, but they do NOT put conda's base \(root\) environment on PATH. To do so, run
>
> $ conda activate
>
> in your terminal, or to put the base environment on PATH permanently, run
>
> $ echo "conda activate" &gt;&gt; ~/.bashrc
>
> Previous to conda 4.4, the recommended way to activate conda was to modify PATH in your ~/.bashrc file. You should manually remove the line that looks like
>
> export PATH="/home/ubuntu/miniconda3/bin:$PATH"
>
> ^^^ The above line should NO LONGER be in your ~/.bashrc file! ^^^

To fix it, do the first thing it recommends and refresh your terminal session:

```bash
echo ". /home/ubuntu/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
```

## Google Colab / Jupyter setup

For users of Google Colab or Jupyter, simply use the Conda environment `lab` as the kernel. For Google Colab, this has to be exposed via the sys path. Run the following:

```bash
%%bash
conda activate lab
# use the location of the lab Conda environment below

%%bash
conda list env

python
import sys
# replace /root/miniconda3/envs/lab with your path from above
sys.path.append('/root/miniconda3/envs/lab/lib/python3.7/site-packages')

print('Python version')
print(sys.version)
```

## `GLIBCXX_3.4.21`version errors due to `gcc, g++, libstdc++`

You encounter libgcc errors like:

> ImportError: /home/deploy/miniconda3/envs/lab/lib/python3.6/site-packages/torch/../../.././libstdc++.so.6: version \`GLIBCXX\_3.4.21' not found \(required by /home/deploy/miniconda3/envs/lab/lib/python3.6/site-packages/ray/pyarrow\_files/pyarrow/lib.cpython-36m-x86\_64-linux-gnu.so\)

Try installing libgcc in Conda:

```text
  conda install libgcc
```

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
./bin/setup
```

## JSON parsing issue in spec

Newer dependencies of SLM Lab may cause issues when parsing JSON spec files. SLM Lab uses a looser JSON syntax which includes comma in the last element of enumerable. If you encounter a JSON parsing issue, simply edit the spec file to remove these extraneous commas.

## Vizdoom installation fails or not found

Manually install it:

```bash
conda activate lab
sudo pip install vizdoom
```

## How to kill stuck processes?

You can see the running processes using tools like [glances](https://github.com/nicolargo/glances). Use the following commands to kill processes by their names. You may need to use `sudo`.

```bash
pkill -f run_lab
pkill -f slm-env
pkill -f ipykernel
pkill -f ray
pkill -f orca
pkill -f Xvfb
ps aux | grep -i Unity | awk '{print $2}' | xargs sudo kill -9
```

## No GUI or images saved on a headless remote server

When running SLM Lab on a remote server, you may get `NoSuchDisplayException: Cannot connect to "None"`. Or your graphs may not be generated. This is because servers are typically headless, i.e. without a display. This error occurs when you're trying to render without a headless display.

First, try setting environment variable `RENDER=false` before the lab command, for example:

```bash
RENDER=false python run_lab.py slm_lab/spec/demo.json dqn_cartpole train
```

Another option is to install Xvfb, and prepend your command with `xvfb-run -a`. For example:

```bash
xvfb-run -a python run_lab.py slm_lab/spec/demo.json dqn_cartpole train
```

## How to forward GUI from a remote server?

If you are running via `ssh` and want GUI forwarding from a server, do:

* [install X11 on your server](https://help.ubuntu.com/community/ServerGUI)
* install OpenGL and/or configure Nvidia driver on your server. [Follow instructions here.](https://github.com/openai/gym/issues/468)
* [install XQuartz/Xming on your laptop](https://uisapp2.iu.edu/confluence-prd/pages/viewpage.action?pageId=280461906)
* do `ssh` with a `-X` flag, e.g. `ssh -X foo@bar`.

## How to sync data from a remote server?

SLM Lab produces a lot of data which are then zipped for our convenience of transferring/syncing them. We use Dropbox to upload these zip files. Follow [this instruction](https://linoxide.com/linux-how-to/install-dropbox-ubuntu/) to install Dropbox CLI.

## What is SLM?

SLM stands for _Strange Loop Machine_, in homage to Hofstadter’s iconic book [_Gödel, Escher, Bach: An Eternal Golden Braid_](https://www.amazon.com/G%C3%B6del-Escher-Bach-Eternal-Golden/dp/0465026567). This lab is created as part of a long term project to try out AI ideas heavily influenced by it.

## Reporting Issues

Can't find the issues you encountered? [Report new issues on Github](https://github.com/kengz/SLM-Lab/issues); it helps all of us.

