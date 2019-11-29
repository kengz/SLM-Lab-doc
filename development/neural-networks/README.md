# Net

## 🏗 Net API

Code: [slm\_lab/agent/net](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/net)

These networks are usable for all algorithms, and the lab takes care of the proper initialization with proper input/output sizing. One can swap out the network for any algorithm with just a spec change, e.g. make `DQN` into `DRQN` by substituting the net spec `"type": "MLPNet"` with `"type": "RecurrentNet"`.

* **MLPNet**
* **RecurrentNet**
* **ConvNet**

These networks are usable for Q-learning algorithms. For more details see [this paper](http://proceedings.mlr.press/v48/wangf16.pdf).

* **DuelingMLPNet**
* **DuelingConvNet**

Reinforcement learning \(RL\) algorithms typically involve approximating one or more unknown, complex, non linear functions. Deep neural networks make good candidates for these function approximators since they excel at approximating complex functions, particularly if the states are characterized by pixel level features.

Neural networks consists of many simple computational operations grouped into sequential layers. For an introduction to neural networks see [this article](https://learningmachinelearning.org/2016/07/). For further reading see [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).

There are a variety of families of neural networks, which organize computation differently and are specialized to different types of tasks. A number of the core neural network families are provided to use out of the box.

* [Multi-layered Perceptron \(MLP\)](mlp.md): Takes a single state as input. These networks are composed of a sequence of dense \(fully connected\) layers. General purpose, simplest network. Well suited for environments with a low dimensional state space.
* [Convolutional Neural Network \(CNN\)](cnn.md): Takes a single state as input. Consists of one or more convolutional layers, followed by one or more dense layers. Excels at image processing. Well suited for environments with pixel level inputs or inputs with a spatial component.
* [Recurrent Neural Network \(RNN\)](rnn.md): Takes a sequence of states as input. Consists of zero or more state processing layers. The output from these layers are passed to a recurrent layers. Specialized to processes sequences. Well suited to environments in which making a decision about how to act in state s would benefit from knowing what states came previously.
* _Pending: Convolutional Recurrent Neural Network \(CRNN\):_ Combines RNN and CNN. Takes a sequence of states as input.

The structure of the inputs and outputs is task \(environment\) and algorithm dependent in RL. This is handled automatically for users. Users need to specify the structure of the hidden network layers, the activation function, and the optimization strategy, through the following parameters.

