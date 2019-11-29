# PrioritizedReplay

## PrioritizedReplay

Code: [slm\_lab/agent/memory/prioritized.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/prioritized.py)

[Prioritized Experience Replay \(PER\)](https://arxiv.org/pdf/1511.05952.pdf) extends from Replay by calculating prioritization for sampling experiences based on errors in Q-values estimation.

Suitable for off-policy algorithms.

### Source Documentation

Refer to the class documentation and example memory spec from the source: [slm\_lab/agent/memory/prioritized.py\#L87-L104](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/prioritized.py#L87-L104)

### **Example Memory Spec**

This specification creates a PrioritizedReplay \(off-policy\) memory with a maximum capacity of 10,000 elements, with a batch size of 32, and CER is disabled. The `alpha` and `epsilon` parameters are specific to PER in computing the errors.

```javascript
{
    ...
    "agent": [{
      "memory": {
        "name": "PrioritizedReplay",
        "alpha": 0.6,
        "epsilon": 0.001,
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": false
      }
    }],
    ...
}
```

For more concrete examples of memory spec specific to algorithms, refer to the existing [spec files](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec).

