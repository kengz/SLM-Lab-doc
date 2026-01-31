# Graphs and Data

SLM Lab automatically generates graphs showing how your agent learns over time. These help you:

- **Track progress**: See if rewards are increasing
- **Compare runs**: Check if different random seeds give consistent results
- **Tune hyperparameters**: Compare different settings to find what works best

## Session, Trial, and Experiment Graphs

SLM Lab produces graphs at each level of the [hierarchy](../using-slm-lab/lab-organization.md):

|                                              Graph                                              |                                              MA Graph                                             |
| :---------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
| ![](../.gitbook/assets/a2c_nstep_breakout_t5_s0_session_graph_train_mean_returns_vs_frames.png) | ![](../.gitbook/assets/a2c_nstep_breakout_t5_s0_session_graph_eval_mean_returns_ma_vs_frames.png) |
|       ![](../.gitbook/assets/a2c_nstep_breakout_t5_trial_graph_mean_returns_vs_frames.png)      |      ![](../.gitbook/assets/a2c_nstep_breakout_t5_trial_graph_mean_returns_ma_vs_frames.png)      |
|     ![](../.gitbook/assets/a2c_nstep_breakout_multi_trial_graph_mean_returns_vs_frames.png)     |     ![](../.gitbook/assets/a2c_nstep_breakout_multi_trial_graph_mean_returns_ma_vs_frames.png)    |

The first row shows the Session graphs. Session is the lowest level, running a single agent with an environment, with the total rewards checkpointed at specified frequency. It is common to smoothen the graph by taking the moving average with a window of 100 checkpoints.

The second row shows the Trial graphs. Trial is a collection of Sessions, and the Trial graph takes the Sessions' mean and plotting an error envelop using standard deviation. This is the most commonly seen deep RL graph as it helps visualize the averaged performance of an instance of a run, repeated over with multiple random seeds.

The last row shows the Experiment (multi trial) graph. An Experiment is a collection of Trials with varying hyperparameters designed to study a hypothesis, and this graph simply overlays the Trial graphs in the experiment for comparison.

## Experiment Variable Graph

Apart from the multi-trial graph, there is another experiment graph which plots **the final performance metrics vs. experiment variables** – the varied hyperparameter values. This helps us see the relationship between the metrics and the experiment variables more clearly. Continuing with the examples above, the experiment variable graph of these trials are plotted below (to view it, download and zoom the graph):&#x20;



![](../.gitbook/assets/a2c_nstep_breakout_experiment_graph.png)

The color of each dot represent the overall performance of the trial, the darker shade the better. Although only one variable (n of n-step) is shown here, if we have multiple variables in an experiment, this graph will contain more columns of subplots corresponding to the experiment variables.

## Experiment Dataframe

In fact, the experiment variable graph is plotted from the experiment dataframe, saved with suffix `experiment_df.csv`. An example of this is shown below:

![](<../.gitbook/assets/experiment df.png>)

This dataframe is another useful piece of output data; it sorts best-trials-first by their performance metric, so it can easily be used to read and find the best-performing trial. The experiment dataframe first lists the trial index, then the experiment variables, and finally a set of predefined metrics.

## Advanced Usage

More graphs are available in the `data/{spec_name}_{timestamp}/graph/` data folder produced from a run, including graphs of loss values, learning rate, etc. These are for more advanced users.
