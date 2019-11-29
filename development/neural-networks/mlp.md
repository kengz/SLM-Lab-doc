# MLP

## Multi-Layer Perceptron

Code: [slm\_lab/agent/net/mlp.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/net/mlp.py)

These networks take a single state as input. They are composed of a sequence of dense \(fully connected\) layers. MLPs are general purpose, simple networks. Well suited for environments with a low dimensional state space, or a state space with no spatial structure.

### Source Documentation

Refer to the class documentation and example net spec from the source: [slm\_lab/agent/net/mlp.py\#L12-L58](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/net/mlp.py#L12-L58)

### **Example Net Spec**

This specification instantiates an MLP with 3 hidden layers of 256, 128, and 64 nodes respectively, rectified linear \(ReLU\) activations, and the [Adam](https://arxiv.org/abs/1412.6980) optimizer with a learning rate of 0.02. The rest of the spec is annotated below.

```javascript
{
    ...
    "agent": [{
      "net": {
        "type": "MLPNet",
        "shared": false,  // whether to shared networks for Actor-Critic
        "hid_layers": [256, 128, 64],
        "hid_layers_activation": "relu",
        "out_layer_activation": null,  // output layer activation
        "init_fn": "xavier_uniform_",  // weight initialization
        "clip_grad_val": 1.0,  // clip gradient by norm
        "loss_spec": {  // default loss function used for regression
          "name": "MSELoss"
        },
        "optim_spec": {  // the optimizer and its arguments
          "name": "Adam",
          "lr": 0.02
        },
        ...
      }
    }],
    ...
}
```

For more concrete examples of net spec specific to algorithms, refer to the existing [spec files](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec).

