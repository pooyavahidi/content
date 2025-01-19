# Neural Networks
Neural Networks are a class of machine learning algorithms that are inspired by the structure of the human brain. This idea started in 1940s to build algorithms that could mimic the human brain.

While Neural Networks have been around for a long time, they resurge in poplularity in 2000s due to the increase in computational power and the availability of large datasets, under new branding as **Deep Learning**

_Artificial Neural Networks (ANN)_ is another name for neural networks which explicitly distinguishes them from biological neural networks (human brain).

![](images/nn_human_brain_cell_inspiration.png)
Source: [Khan Academy - Overview of neuron structure and function](https://www.khanacademy.org/science/biology/human-biology/neuron-nervous-system/a/overview-of-neuron-structure-and-function)

Although the structure of a neural network is inspired by the human brain, not only it's a very simplified version of it, but also we still have a very little understanding of how the human brain works. So, it's good to know the origin and similarities, but we should see the neural networks as a mathematical and engineering model rather than the biological motivation behind neural networks.

**Performance of ML Algorithms with respect to Data Size:**<br>
During time as the amount of data available to train models has increased, AI researchers have found that the performance of traditional algorithms (Linear Regression, Logistic Regression, Decision Trees, etc.) plateaus after a certain amount of data. In other words, the performance of these algorithms does not improve significantly after a certain amount of data is provided to them. After that point, no matter how much more data we provide, the performance of these algorithms does not improve significantly.

However, the performance of Neural Networks continues to improve with more data.

![](images/ml_alg_performance_with_data_size.png)

So, if we have a large amount of data, a large neural network can effectively learn from it and provide the best possible performance in comparison to other ML algorithms.

This capability of Neural Networks comes from the fact, that Neural Networks can have many layers of neurons, and more neurons mean more parameters to learn. This allows them to learn complex patterns from the large amount of data.

We also should note that training large neural networks with large datasets requires a lot of computational power. So, both the increase in size of datasets and computational power (GPUs, etc) have contributed to the resurgence of Neural Networks in recent decades, which will be continued as both of these factors continue to grow.


## Structure of a Neural Network

![](images/neural_network_structure.png)

**Neurons**:<br>
Think of a neuron as a simple function that takes some inputs, does some calculations based on an internal function, and produces an output.

Neurons are also called **Activation Units**. The output of a neuron (which is the result of the internal function) is called the **activation value** of that neuron and denoted as $a$.

> The term **activation** inspired by the term activation of human brain neurons.


The internal function of a neuron is also called **Activation Functions**. Some of the most common activation functions are:
- Sigmoid Function
- ReLU (Rectified Linear Unit) Function
- Tanh Function
- Softmax Function


**Layers:**<br>
Neural Network is a collection of layers of neurons. Each layer is a collection of neurons that are connected to the neurons in the previous layer. The first layer is called the **Input Layer** and the last layer is called the **Output Layer**. The layers between the input and output layers are called **Hidden Layers**.

> The term **hidden** inspired by the fact that these layers are not directly connected to the input or output of the network. When we have a training dataset, we can observe both input $X$ and output $Y$ of the dataset. But we can't observe the output of the hidden layers in the training set. So, they are called hidden layers.

In a semantic way, each neuron represents a **feature**.
- If a neuron is in the input layer, it represents an input feature.
- If a neuron is in hidden layers, it represents a learned feature.
- If a neuron is in the output layer, it represents an output class.

A layer can have one or more neurons. The number of neurons in the input layer is determined by the number of input features in the dataset. The number of neurons in the output layer is determined by the number of classes which we want to predict.

In a simplest form, a neural network can have only one layer with one neuron.

Think of a neural networks of multiple layers which each layer inputs a vector of features and outputs another vector of features. The output of the previous layer is the input of the next layer.

**Depth and Width**: The number of layers in a neural network is called the **depth** of the network. The more layers, the deeper the network. The number of neurons in a layer is called the **width** of the network. The more neurons, the wider the network.

**Neural Network Architecture**: When creating a neural network, we should decide the depth and width (number of layers and number of neurons in each layer) of the network. This is called **architecture** of the network. The architecture of the network is a hyperparameter that we should decide before training the network.

**Connections:**<br>
Each neuron in a layer is connected to all neurons in the previous layer. These connections are called **edges**. Each edge has a **weight** which is a parameter that the neural network learns during training.

This weight is multiplied by the output of the previous neuron. If model tries to reduce the impact of a neuron, it can reduce the weight of the connection between that neuron and the current neuron.

> The term **Perceptron** is original name for one neuron (one computational unit) which was introduced in 1950s. The term **Perceptron** is still used in some contexts to refer to a single neuron (a single activation unit). The term **Multi-Layer Perceptron (MLP)** is used to refer to a neural network with multiple layers of neurons.


### Each Layer Learns From the Previous Layer
This is the key reason why neural networks are so capable and can learn complex patterns from the data. Because they can learn **new** features from the input features, and then learn **new** features from those learned features, and so on. The deeper (more layers) the neural network, the more new learned features it can have, so the more complex patterns it can learn.

**Learned features** in hidden layers are the features that neural network learns on its own from the learned features of the previous layer. The first hidden layer learns features from the input features. The second hidden layer learns from the **learned features** of the first hidden layer, and so on. So, each layer learns higher level features and more complex patterns from the learned features of the previous layer.


This is the key difference comparing to othe ML algorithms which the model is limited to the features that we provide to it. As we discussed in [feautre engineering](feature_engineering.md), we can engineer new features from the existing features, but this is a manual process and in many cases we can't engineer all possible relevant features. Also, we could see that engineering new features can cause problems like [curse of dimensionality](feature_engineering.md#dimensionality-reduction) and [overfitting](generalization.md#overfitting). But in neural networks, the model can learn new features automatically and much more efficiently than manual feature engineering.


**Each layer learns higher level features**<br>
Let's say we want to classify images of persons. An image is a matrix of pixels. For example for a $100 \times 100$ pixel image, we have $10,000$$ pixels. If the image is grayscale (black and white), each pixel has a value between 0 and 255 which represents the _intensity_ of the pixel (0 is the complete black and 255 is the complete white, and all numbers between are shades of gray). So, this image can be represented as a $100\times100$ matrix of scalar values between 0 and 255. We can represent this matrix as a vector of $10,000$ values.

![](images/nn_image_input_features_as_vector.png)

$$
\vec{\mathbf{x}} = \begin{bmatrix} x_1\\
x_2 \\
\vdots \\
x_{10000}
\end{bmatrix}
= \begin{bmatrix} 38\\
231 \\
\vdots \\
85
\end{bmatrix}
$$

where:
- $x_i$ is the intensity of the $i^{th}$ pixel. $x_1$ is the first pixel in the first row, $x_2$ is the second pixel in the first row, $x_{101}$ is the first pixel in the second row, and so on.


> Most images use the RGB (Red, Green, Blue) color model, where the color of each pixel is determined by the combination of three color channels: red, green, and blue. Each channel has a value ranging from 0 to 255, where 0 represents the lowest intensity and 255 the highest intensity. So, if our example image was colored, we would have 3 channels for each pixel, and the image would be represented as a $100x100x3$ matrix of scalar values.

**Each layer learns from learned-features of the previous layer**<br>
The first layer of the neural network can learn features like edges, corners, and textures from the input image. The second layer can learn features like shapes, objects, and patterns from the learned features of the first layer. The third layer can learn features like faces, objects, and scenes from the learned features of the second layer. And so on.

![](images/nn_layers_learning_higher_level_features.png)

Source: [Convolutional Deep Belief Networks
for Scalable Unsupervised Learning of Hierarchical Representations](https://web.eecs.umich.edu/~honglak/icml09-ConvolutionalDeepBeliefNetworks.pdf)



Further reading here: [Understanding Neural Networks Through Deep Visualization](https://arxiv.org/abs/1506.06579)
