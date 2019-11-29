# Motivation

## Why SLM Lab was created

**SLM Lab** was developed as the authors started investigating ideas in deep RL. The field is very empirical, and tools were needed to do deep RL like experimental science.

We quickly found that there are so many moving parts that it is difficult to build intuition or test things systematically. It is also difficult for newcomers to get started - there is a reasonably high burden of basic knowledge required.

We wanted to be able to work on DRL using the workflow of experimental science. There was a need for a framework that would allow us to compare algorithms and environments, quickly set up experiments to test hypotheses, reuse components, analyze and compare results, log results. We also wanted to avoid dealing with tons of command line arguments, bash scripts and manual search, or trawling through log files.

The tools we built to do this became the SLM Lab. Most importantly, the unified framework helps to minimize hidden side effects or forgotten details that contribute to irreproducibility and results that are not comparable. And much like how science is communicated with standard units such as kilograms or meters, we created metrics within the lab for the same purpose too.

Now, the lab’s **workflow** goes as:

* have a hypothesis, say to investigate the effects of regularization in RL
* implement new components if any \(usually none/little; there is a lot of reuse\)
* set up an experiment using a simple JSON spec file
* run it on a server
* get the results and report it back with a Pull Request

Given this, SLM Lab has a lot of elements in common with other libraries \(search, evaluation metrics, baselines, etc.\) but is the first to bring them together in one place. This allows us to **design, run, measure, and record experiments end-to-end**.

SLM-Lab has the same two purposes that science labs have: research and education. A few immediate goals of the lab are:

* **easier and faster** development cycle through reusable components. Many RL folks we know spends too much time debugging and building auxiliary components from scratch, instead of focusing on just the research.
* manual experiment design, parameter search and evaluation is extremely slow; the lab **automates away a lot of these time-consuming things**.
* **address the reproducibility problem** in DRL research by standardizing implementation, benchmarking, and adding rigor to evaluation. Also the full config data is committed to the log book.
* it can be **used to teach** tutorials/classes, and lowers the barrier of entry to DRL, just like how experiments can be used to teach science more vividly. Plus, it is very much still an experimental science.

## Design Principles

SLM Lab is created for deep reinforcement learning research and applications. The design was guided by four principles:

#### Modularity

* makes research easier and more accessible: reuse well-tested components and only focus on the relevant work
* makes learning deep RL easier: the algorithms are complex; SLM Lab breaks them down into more manageable, digestible components
* components get reused maximally, which means less code, more tests, and fewer bugs

#### Simplicity

* the components are designed to closely correspond to the way papers or books discuss RL
* modular libraries are not necessarily simple. Simplicity balances modularity to prevent overly complex abstractions that are difficult to understand and use

#### Analytical clarity

* hyperparameter search results are automatically analyzed and presented hierarchically in increasingly granular detail
* it should take less than 1 minute to understand if an experiment yielded a successful result using the [experiment graph](../analyzing-results/session-graph.md)
* it should take less than 5 minutes to find and review the top 3 parameter settings using the [trial and session graphs](../analyzing-results/session-graph.md)

#### Reproducibility

* only the spec file and a git SHA are needed to fully reproduce an experiment
* all the results are recorded in the [Benchmark Result](../benchmark-results/discrete-benchmark.md) pages
* experiment reproduction instructions are submitted to the Lab via [`result` Pull Requests](https://github.com/kengz/SLM-Lab/pulls?utf8=%E2%9C%93&q=is%3Apr+label%3Aresult+)
* the full experiment datas contributed are public on Dropbox 

