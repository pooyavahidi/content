# Neural Networks Layers
As part of any neural network, the layers are the key building blocks of its architecture. The different types with different properties are used in different tasks to achieve the desired results. The most common layers are:

## Fully Connected Layer (Dense)
This is the most common type of layer in neural networks. The **dense** layer connect all the neurons from the _previous_ layer to the its neurons by weights.

Every feature in the input affects every neuron in the output (controlled by the weights of that connection). Each neuron first perform the **linear transformation** (weighted sum of it's inputs) and then apply an **activation function** to the result.

Fully Connected means there are no skipped connections—every input influences every output. This is different from convolutional layers, where connections are local (each neuron only looks at a small region of the input).

![](images/nn_fully_connected_layer.png)

As we can see every neuron in this layer is a function of the output of all the neurons from the _previous_ layer.

In this example the layer 2 is a fully connected layer. So, the activation value of the first neuron is a function of all the activation values of the previous layer $\vec{\mathbf{a}}^{[1]}$.

$$a^{[2]}_1 = g(\vec{\mathbf{w}}_1^{[2]} \cdot \vec{\mathbf{a}}^{[1]} + b^{[2]}_1)$$
