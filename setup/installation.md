# Installation

## 💽 Installing SLM Lab

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
For users of Google Colab or Jupyter, simply use the Conda environment `lab` as the kernel. See the [Help](https://slm-lab.gitbook.io/slm-lab/resources/help#google-colab-jupyter-setup) page for more.
{% endhint %}

{% hint style="info" %}
Readers of the book Foundations of Deep Reinforcement Learning: please see [this custom instruction page](../publications-and-talks/instruction-for-the-book-+-intro-to-rl-section.md).
{% endhint %}

## 🖥 Hardware Requirements

Non-image based environments can run on a laptop. Only image based environments such as the Atari games benefit from a GPU speedup. For these, we recommend 1 GPU and at least 4 CPUs. This can run a single Atari `Trial` consisting of 4 `Sessions`.

For desktop, a reference spec is GTX 1080 GPU, 4 CPUs above 3.0 GHz, and 32 GB RAM.

For cloud computing, start with an affordable instance of [AWS EC2 `p2.xlarge`](https://aws.amazon.com/ec2/instance-types/p2/) with a K80 GPU and 4 CPUs. Use the Deep Learning AMI with Conda when [creating an instance](https://aws.amazon.com/getting-started/tutorials/get-started-dlami/). 

