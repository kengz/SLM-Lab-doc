# Installation

## :minidisc: Installing SLM Lab

Clone the repository:

```bash
git clone https://github.com/kengz/SLM-Lab.git
```

Install the dependencies:

```bash
cd SLM-Lab/
./bin/setup
```

This runs a prepared bash script with the necessary setup steps, with Python dependencies managed through Conda. Refer to the [Help](../resources/help.md) page if you encounter issues.

{% hint style="info" %}
Readers of the book:book:_Foundations of Deep Reinforcement Learning:_ please see [this custom instruction page](../publications-and-talks/instruction-for-the-book-+-intro-to-rl-section.md).
{% endhint %}

### Alternative Installations

#### Bring-your-own-PyTorch/GPU

SLM Lab uses PyTorch version 1.3.1 by default. For newer GPU cards, the versions of PyTorch and CUDA that come with SLM Lab default setup above may not be supported. In this case, you may bring-your-own-PyTorch by using [environment-byo.yml](https://github.com/kengz/SLM-Lab/blob/master/environment-byo.yml) to setup dependencies instead. This installs the same Python modules except for PyTorch and cudatoolkit. Use the following commands:

```bash
# first install the system dependencies
sudo apt-get update && \
  apt-get install -y build-essential \
  curl nano git wget zip libstdc++6 \
  python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb && \
  rm -rf /var/lib/apt/lists/*

# setup Conda environment and install everything except PyTorch and cudatoolkit
conda create -n lab python=3.7.3 -y
conda env update -f environment-byo.yml
# install your own pytorch from https://pytorch.org/get-started/locally/
conda activate lab
conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch
```

Thanks to [**@Karl-Grantham**](https://github.com/Karl-Grantham) for help with testing this on RTX A6000 and RTX A5000 GPUs.

#### Windows

The best way to run SLM Lab on Windows is to use a Bash shell/Linux subsystem. Credit to [**@vladimirnitu**](https://github.com/vladimirnitu) and [**@steindaian**](https://github.com/steindaian) for providing a detailed instruction PDF for doing so on Windows:

{% file src="../.gitbook/assets/SLM_for_Windows.pdf" %}
SLM Lab for Windows (Instruction PDF)
{% endfile %}

#### Google Colab/Jupyter Notebook

Although we do not recommend running SLM Lab on Google Colab or Jupyter notebooks (notebooks come with inherent limitations, e.g. no rendering/multi-processing), we have prepared an example notebook for illustration. Credit to [**@piosif97**](https://github.com/piosif97) for helping with this:

{% hint style="info" %}
[SLM Lab Colab notebook](https://gist.github.com/kengz/6fd52a902129fb6d4509c721d71bda48)
{% endhint %}

For details on how it works, refer to [this Help section](https://slm-lab.gitbook.io/slm-lab/resources/help#google-colab-jupyter-setup).

#### Google Colab (updated 2024)

Another user [**@isacciobota**](https://github.com/isacciobota) has figured out installation in Colab in 2024 - an especially helpful update given the dependencies/hardware architecture changes in the past years. Please find their detailed instructions with additional examples here:

{% hint style="info" %}
[SLM Lab Colab notebook (2024)](https://github.com/isacciobota/SLMLab-x-GoogleColab)
{% endhint %}

## :desktop: Hardware Requirements

Non-image based environments can run on a laptop. Only image based environments such as the Atari games benefit from a GPU speedup. For these, we recommend 1 GPU and at least 4 CPUs. This can run a single Atari `Trial` consisting of 4 `Sessions`.

For desktop, a reference spec is GTX 1080 GPU, 4 CPUs above 3.0 GHz, and 32 GB RAM.

For cloud computing, start with an affordable instance of [AWS EC2 `p2.xlarge`](https://aws.amazon.com/ec2/instance-types/p2/) with a K80 GPU and 4 CPUs. Use the Deep Learning AMI with Conda when [creating an instance](https://aws.amazon.com/getting-started/tutorials/get-started-dlami/).&#x20;
