# CNN

## Convolutional Neural Network

Code: [slm\_lab/agent/net/conv.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/net/conv.py)

These networks take a single state as input and produce one or more outputs. It consists of zero or more convolutional layers, followed by zero or more dense layers. Finally, there are one or more dense output layers. CNNs excel at image processing and processing inputs with 2D or 3D structure. This makes them well suited for environments with pixel level inputs or inputs with a spatial component.

For more information on each of the elements see this [tutorial](https://github.com/lgraesser/CNN-Tutorial) on CNNs or the [PyTorch documentation](http://pytorch.org/docs/0.3.0/nn.html#convolution-layers)

### Source Documentation

Refer to the class documentation and example net spec from the source: [slm\_lab/agent/net/conv.py\#L11-L75](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/net/conv.py#L11-L75)

### **Example Net Spec**

This specification instantiates a typical CNN for deep RL with 3 convolutional layers \(input channels are automatically inferred\):

* 32 output channels, kernel size \(8 x 8\), stride of 4, padding of 0, and dilation of \(1 x 1\)
* 64 output channels, kernel size \(4 x 4\), stride of 2, padding of 0, and dilation of \(1 x 1\)
* 32 output channels, kernel size \(3 x 3\), stride of 1, padding of 0, and dilation of \(1 x 1\)

Then, the output of the last convolutional layer is flattened and connected to the fully-connected hidden layers \(this can be empty\):

* a layer of 512 units, then to a layer of 256 units

Lastly, the final layer is connected to an output layer that is automatically constructed by inferring the the algorithm and the environment.

The network uses linear \(ReLU\) activations, no activation for the output layer, and no weight initialization function. The rest of the spec is annotated below.

```javascript
{
    ...
    "agent": [{
      ...
      "net": {
        "type": "ConvNet",
        "shared": false,  // whether to shared networks for Actor-Critic
        "conv_hid_layers": [
            [32, 8, 4, 0, 1],
            [64, 4, 2, 0, 1],
            [32, 3, 1, 0, 1]
        ],
        "fc_hid_layers": [512, 256],
        "hid_layers_activation": "relu",
        "out_layer_activation": null,
        "init_fn": null,  // weight initialization
        "normalize": false,  // whether to divide input by 255.0
        "batch_norm": false,  // whether to add batchnorm layers
        "clip_grad_val": 1.0,  // clip gradient by norm
        "loss_spec": {
          "name": "SmoothL1Loss"  // default loss function used for regression
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

