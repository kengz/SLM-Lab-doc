# Algorithm

## 🏗 Algorithm API

Code: [slm\_lab/agent/algorithm](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/algorithm)

Algorithm is the main class which implements an RL algorithm. This includes declaring its networks and variables, acting, sampling from memory, and training. It initializes its networks and memory by simply calling the [Memory](../memory/) and [Net](../neural-networks/) classes with their specs. The loss functions for the algorithms is also implemented here.

Each algorithm comes with a number of hyperparameters that can be specified through a [agent spec file](https://github.com/kengz/slm-lab/tree/d4af0388c41fee09d3266973e4c791c11b144566/baselines/usage/agent-spec.md).

## **Algorithm Spec**

```javascript
{
    ...
    "agent": [{
      "name": str,
      "algorithm": {
        "name": str,
        "action_pdtype": str,
        "action_policy": str,
        "gamma": float,
        ...,
      },
      ...
    }],
    ...
}
```

* **name:** name of an implemented algorithm class. This must be a class that conforms to the [algorithm api](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/base.py) and is saved in a `.py` file under [slm\_lab/agent/algorithm](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/algorithm)
* **action\_pdtype:** specifies the probability distribution that actions are sampled from. For example, "Argmax" or "Categorical" for discrete action spaces, or "Normal", "MultivariateNormal", and "Gumbel" for continuous action spaces. These are declared in [slm\_lab/agent/algorithm/policy\_util.py\#L18-L24](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/policy_util.py#L18-L24)
* **action\_policy:** specifies how the agent should act. e.g. "epsilon\_greedy". These are declared in [slm\_lab/agent/algorithm/policy\_util.py\#L133](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/policy_util.py#L133)
* **gamma** $$\in[0,1]$$ how much to discount the future for the returns. 0 corresponds to complete myopia, the agent only cares about the current time step. 1 corresponds to no discounting. Each future state matters as much as the current state.

Other algorithm spec hyperparameters are specific to algorithm implementations. For those, refer to the class documentation of algorithms in [slm\_lab/agent/algorithm](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/algorithm).

For more concrete examples of algorithm spec specific to algorithms, refer to the existing [spec files](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec).

To learn more about algorithms, check out [Deep RL Resources](../../resources/untitled.md).

{% hint style="info" %}
The subpages to follow showcase a subset of algorithms in SLM Lab. See [here](../../#algorithms) for the list of implemented algorithms in SLM Lab.
{% endhint %}

