# TensorBoard: Visualizing Models and Actions

## :chart\_with\_upwards\_trend: Built-in TensorBoard

TensorBoard is built-in to SLM Lab. It records all metric variables logged to the terminal, plus the PyTorch model graph, model parameter histograms, and action histograms. All of these are done automatically during checkpointing. This allows for richer diagnosis of the network and policy, e.g. by seeing if the distributions shift over the course of learning.

TensorBoard event files are saved to the `log/` folder in the output data. During/after a run, you can launch TensorBoard:

```bash
uv run tensorboard --log_dir=data
```

{% hint style="info" %}
It may take time for TensorBoard to parse the event files. Speed it up by providing a specific folder, e.g. `--log_dir=data/ppo_bipedalwalker_2024_01_15_123456/log`.
{% endhint %}

Then, go to `localhost:6006` on your browser, and you should see the TensorBoard page:

![](https://user-images.githubusercontent.com/8209263/66803221-d9bc0980-eed3-11e9-92b8-0e5cd42a6eab.png)

The histogram tab is useful for revealing the distributions of the actions. In the example above, BipedalWalker has 4 continuous actions, hence there are 4 groups for plots for visualizing the value distributions of these 4 actions across different trials and sessions. Likewise, all the model parameters of an agent is also recorded as value distributions of the parameters of their layers.

In the histograms, the vertical axis (coming out from the page) is the number of frames during checkpoints. As an agent learns over time, we should see the distributions changing in shape and shifting locations.
