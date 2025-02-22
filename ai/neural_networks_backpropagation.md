# Neural Network Backpropagation
Backpropagation is an algorithm to calculate the [derivative (gradient)](../math/derivatives.md) of the cost function with respect to the parameters of the neural network.

The name of the Backpropagation which is also called _back prop_ or _backward pass_, coming from the fact that after the [forward pass](neural_networks_inference.md) which calculates the output of the network based on the current parameters, then backprop calculates the derivative (gradient) of the cost function with respect to the parameters in the reverse order of the forward pass, meaning from the output layer back to the first layer. Hence, the name back propagation.
