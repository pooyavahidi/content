# Neural Networks Training
Similar to training a [linear regression model](regression.md), training a neural network involves finding the optimal weights and biases that minimize the error between the predicted output and the actual output. The steps of training a neural network are very similar to those of training a linear regression model:

1. **Define the model**: Specify the neural network model $f_{W,B}(X)=?$ by defining the input, output and internal architecture of the network:
   - The [number and types of layers](neural_networks_layers.md) and the number of neurons in each layer.
   - The [activation functions](neural_networks_activation_functions.md) in each layer, and particularly the activation function of the output layer based on the type of the problem (regression, binary classification, or multi-class classification, etc).
   - The [loss function](loss_and_cost_functions.md) according to output layer.

2. [**Gradient Descent**](gradient_descent.md): This is the optimization algorithm to minimize the error between the predicted output and the actual output by updating the weights and biases step by step. See the steps of the gradient descent algorithm in the [gradient descent algorithm](gradient_descent.md#gradient-descent-algorithm).

   The key components of gradient descent is calculating the _partial derivatives_ of the loss function with respect to the weights and biases of the network. In neural network this alogrithm is called _backpropagation_.
