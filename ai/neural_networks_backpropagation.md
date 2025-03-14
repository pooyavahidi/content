# Neural Network Backpropagation
Backpropagation is an algorithm to calculate the [derivative (gradient)](../math/derivatives.md) of the cost function with respect to the parameters of the neural network.

The name of the Backpropagation which is also called _back prop_ or _backward pass_, coming from the fact that after the [forward pass](neural_networks_inference.md) which calculates the output of the network based on the current parameters, then backprop calculates the derivative (gradient) of the cost function with respect to the parameters in the reverse order of the forward pass, meaning from the output layer back to the first layer. Hence, the name back propagation.


## Computational Graph
Computation graph is break down of the neural network (forward propagation and back propagation) into smaller building blocks, which then can be calculated step by step. This graph is a directed acyclic graph (DAG), and output of each node is calculated based on the output of the previous nodes. So, it can reuse the output of the previous nodes to calculate the output of the next nodes, which makes the computation more efficient.

In case of backpropagation, the computation graph is like breaking down the chain rule of the derivative into each composite function, and then calculate from the most outer function (cost function) to the most inner function (the parameters of the neural network).


## Resources
- [Google ML Course - Backpropagation](https://developers.google.com/machine-learning/crash-course/backprop-scroll)
