# OnPolicyReplay

## OnPolicyReplay

Code: [slm\_lab/agent/memory/onpolicy.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/onpolicy.py)

OnPolicyReplay differs from the plain Replay in that when `memory.sample()`is called in a training step, it returns all of its stored elements, then flushes its storage clean. By default, on-policy training happens at the end of every episode.

Suitable for on-policy algorithms.

### Source Documentation

Refer to the class documentation and example memory spec from the source: [slm\_lab/agent/memory/onpolicy.py\#L14-L35](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/memory/onpolicy.py#L14-L35)

### **Example Memory Spec**

The spec for this memory has no parameters, since it automatically flushes at the end of an episode after training.

```javascript
{
    ...
    "agent": [{
      "memory": {
        "name": "OnPolicyReplay"
      }
    }],
    ...
}
```

For more concrete examples of memory spec specific to algorithms, refer to the existing [spec files](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec).

