# RNN

## Recurrent Neural Network

Code: [slm\_lab/agent/net/recurrent.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/net/recurrent.py)

These networks take a sequence of states as input and produce one or more outputs. They consist of zero or more state processing layers \(organized as an MLP\). All of the states are passed through the MLP \(if there is one\) and the transformed states, are passed in sequence to the recurrent layer. RNNs are structured so as to retain information about a sequence of inputs. This makes them well suited to environments in which making a decision about how to act in state `S` would benefit from knowing which states came previously.

### Source Documentation

Refer to the class documentation and example net spec from the source: [slm\_lab/agent/net/recurrent.py\#L10-L71](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/net/recurrent.py#L10-L71)

### **Example Net Spec**

This specification instantiates a RecurrentNet with two components. First a state processing MLP with with 2 hidden layers of 256 and 128 nodes respectively and rectified linear \(RELU\) activations. This is followed by one recurrent [GRU](https://arxiv.org/abs/1409.1259) layer with a hidden state of 64 units. The optimizer is [Adam](https://arxiv.org/abs/1412.6980) with a learning rate of 0.01. The number of sequential states used as input to the networks is 4. The rest of the spec is annotated below.

```javascript
{
    ...
    "agent": [{
      "net": {
        "type": "RecurrentNet",
        "shared": false,  // whether to shared networks for Actor-Critic
        "cell_type": "GRU",
        "fc_hid_layers": [256, 128],
        "hid_layers_activation": "relu",
        "out_layer_activation": null,
        "rnn_hidden_size": 64,
        "rnn_num_layers": 1,
        "bidirectional": false,  // whether to use bidirectional layer
        "seq_len": 4,
        "init_fn": "xavier_uniform_",  // weight initialization
        "clip_grad_val": 1.0,  // clip gradient by norm
        "loss_spec": {  // default loss function used for regression
          "name": "MSELoss"
        },
        "optim_spec": {  // the optimizer and its arguments
          "name": "Adam",
          "lr": 0.01
        },
        ...
      }
    }],
    ...
}
```

For more concrete examples of net spec specific to algorithms, refer to the existing [spec files](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec).

