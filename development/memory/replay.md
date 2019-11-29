# Replay

## Replay

Code: [master/slm\_lab/agent/memory/replay.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/replay.py)

Experiences are stored in a circular buffer which grows until the memory has reached capacity. Once the memory is at capacity the oldest experiences are deleted to make space for the newest.

Batches of size `batch_size` are sampled from the entire memory. Sampling is random uniform unless "PrioritizedReplay" memory is used.

Suitable for off-policy algorithms.

### Source Documentation

Refer to the class documentation and example memory spec from the source: [slm\_lab/agent/memory/replay.py\#L43-L67](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/replay.py#L43-L67)

### **Example Memory Spec**

This specification creates a Replay memory with a maximum capacity of 10,000 elements, i.e. it can store experiences for 10,000 time steps. When `memory.sample()` is called, it will return a batch of 32 elements. We also specify the use of CER \(Combined Experience Replay\), which guarantees the latest experience is included in the samples.

```javascript
{
    ...
    "agent": [{
      "memory": {
        "name": "Replay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
      }
    }],
    ...
}
```

For more concrete examples of memory spec specific to algorithms, refer to the existing [spec files](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec).

