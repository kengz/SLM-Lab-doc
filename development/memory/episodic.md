# OnPolicyBatchReplay

## **OnPolicyBatchReplay**

Code: [slm\_lab/agent/memory/onpolicy.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/onpolicy.py)

Sometimes for on-policy training, an agent trains at a fixed frequency, e.g. every 200 steps, instead of at the end of an episode. The signal for training also comes from the memory class in `memory.to_train`. For batched training schedule, OnPolicyBatchReplay extends OnPolicyReplay to set `self.to_train = true`  based on agent's training frequency and the number of experiences collected.

Suitable for on-policy algorithms.

### Source Documentation

Refer to the class documentation and example memory spec from the source: [slm\_lab/agent/memory/onpolicy.py\#L100-L110](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/onpolicy.py#L100-L110)

### **Example Memory Spec**

The spec for this memory has no parameters, since it automatically flushes at the end of a batch after training. The `batch_size` is the `training_frequency` provided by algorithm\_spec.

```javascript
{
    ...
    "agent": [{
      "memory": {
        "name": "OnPolicyBatchReplay"
      }
    }],
    ...
}
```

For more concrete examples of memory spec specific to algorithms, refer to the existing [spec files](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec).

