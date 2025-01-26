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
Think of a neuron as a simple function that takes some inputs, does some calculations and produces an output. Neurons sometimes called **nodes** or **units**.

Inspired by the human brain neurons, the following terms are used in neural networks, where a neuron is a computational unit and generate an activation which send down to the other downstream neurons.
- **Activation Unit**: A neuron.
- **Activation**: The output of a neuron denoted as $a$. It also called _activation value_ or _activation output_.
- **Activation Function**: The internal function of a neuron that calculates the output of the neuron.

All the inputs to a neuron are multiplied by some weights (parameters of the neuron), summed up to calculate the _logit_ of the neuron, and then passed through an **activation function** to produce the output of the neuron.



Some of the most common activation functions are:
- ReLU (Rectified Linear Unit) Function
- Sigmoid Function
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

Think of a neural networks of multiple layers which each layer inputs a vector of features and outputs another vector of features. The output of the previous layer is the input of the next layer.

**Depth and Width**: The number of layers in a neural network is called the **depth** of the network. The more layers, the deeper the network. The number of neurons in a layer is called the **width** of the network. The more neurons, the wider the network.

**Neural Network Architecture**: When creating a neural network, we should decide the depth and width (number of layers and number of neurons in each layer) of the network. This is called **architecture** of the network. The architecture of the network is a hyperparameter that we should decide before training the network.

**Connections:**<br>
Each neuron in a layer is connected to all neurons in the previous layer. These connections are called **edges**. Each edge has a **weight** which is a parameter that the neural network learns during training.

During training, model tries to learn the best weights for the connections between neurons. The weight of a connection determines the impact of the neuron (feature or learned feature) on the output of the current neuron. The larger the weight, the larger the impact of the feature on the output of the neuron.

> The term **Perceptron** is original name for one neuron (one computational unit) which was introduced in 1950s. The term **Perceptron** is still used in some contexts to refer to a single neuron (a single activation unit). The term **Multi-Layer Perceptron (MLP)** is used to refer to a neural network with multiple layers of neurons.



### Inside the Neurons of a Layer
Let dive a bit deeper into the internal structure of a neurons in a layer.

![](images/nn_neuron_activation_function.png)

**Notation For Layers and Neurons**:<br>
To denote the layers and neurons in a neural network, we use the following notation:

$$z^{[layer]}_{neuron} \quad \vec{\mathbf{w}}^{[layer]}_{neuron} \quad b^{[layer]}_{neuron} \quad a^{[layer]}_{neuron}$$

Where:
- Superscript $[layer]$ is the number of the layer.
- Subscript $neuron$ is the number of the neuron in the layer.

**logit**:<br>
The weighted sum of the inputs to the neuron is called logit of the neuron. The logit is the input of the activation function. The logit of the neuron $n$ in the layer $l$ is denoted as:

$$z^{[l]}_{n}$$

For example, Logit of the second neuron in the first layer is denoted as:

$$z^{[1]}_2$$


The logit of the first neuron in the first layer is calculated as:

$$z^{[1]}_1 = w^{[1]}_{11} \cdot x_1 + w^{[1]}_{12} \cdot x_2 + b^{[1]}_1$$

Where:

- $w^{[1]}_{11}$ is the weight for input feature $x_1$ to node 1 of layer 1.
- $w^{[1]}_{12}$ is the weight for input feature $x_2$ to node 1 of layer 1.
- $b^{[1]}_1$ is the bias term for node 1 of layer 1. We have only one bias term for each neuron in the layer.

We can write the above as:

$$z^{[1]}_1 = \sum_{i=1}^{n} w^{[1]}_{1i}x_i + b^{[1]}_1$$

> Weight and bias terms for a neuron are denoted as:
> $$w^{[l]}_{ji} \quad \text{and} \quad b^{[l]}_j$$
> Where:
> - $[l]$ indicates the layer number.
> - $j$ indicates the neuron number in the current layer.
> - $i$ indicates the input feature (the neuron number from the previous layer).
>
> Note: We don't have $i$ for the bias term because the bias term is not connected to any neuron in the previous layer. It's just a constant term that is added to the weighted sum of the inputs.

We can also write the logit in a simpler form as a dot product of the weight vector and the input feature vector.

$$z^{[1]}_1 = \vec{\mathbf{w}}_1^{[1]} \cdot \vec{\mathbf{x}} + b^{[1]}_1$$

Where:
- $\vec{\mathbf{w}}_1^{[1]}$ is the vector of weights for input features to the first neuron of the first layer.

**Activation of a Neuron**:<br>

The activation of a neuron is defined as:

$$a^{[layer]}_{neuron} = f(z^{[layer]}_{neuron})$$

So, the activation of the first neuron in the first layer is:

$$a^{[1]}_1 = f(z^{[1]}_1)$$

where:
- $[1]$ is the index of the layer, i.e. layer 1.
- subscript $1$ is the index of the neuron in the layer, i.e. neuron 1.

For example, if we choose Sigmoid Function as the activation function, the activation of the first neuron in the first layer is:

$$a^{[1]}_1 = \sigma(z^{[1]}_1)$$


So, the activation $a^{[1]}_1$ of the first neuron in the first layer is calculated as:

$$a^{[1]}_1 = \sigma(z^{[1]}_1) = \sigma(\vec{\mathbf{w}}_1^{[1]} \cdot \vec{\mathbf{x}} + b^{[1]}_1)$$

Which we can write the sigmoid function explicitly as:

$$a^{[1]}_1 = \frac{1}{1 + e^{-(\vec{\mathbf{w}}_1^{[1]} \cdot \vec{\mathbf{x}} + b^{[1]}_1)}}$$



**Output of a Layer**:<br>
The output of a layer $\vec{\mathbf{a}}^{[layer]}$ is a vector of activation values of the neurons in that layer. For example, the output of the first layer is:

$$\vec{\mathbf{a}}^{[1]} = \begin{bmatrix} a^{[1]}_1\\
a^{[1]}_2 \\
\vdots \\
a^{[1]}_n
\end{bmatrix}$$

where:
- $a^{[1]}_n$ is the activation value of the $n^{th}$ neuron in the first layer.

### Each Layer Learns From the Previous Layer
This is the key reason why neural networks are so capable and can learn complex patterns from the data. Because they can learn **new** features from the input features, and then learn **new** features from those learned features, and so on. The deeper (more layers) the neural network, the more new learned features it can have, so the more complex patterns it can learn.

**Learned features** in hidden layers are the features that neural network learns on its own from the learned features of the previous layer. The first hidden layer learns features from the input features. The second hidden layer learns from the **learned features** of the first hidden layer, and so on. So, each layer learns higher level features and more complex patterns from the learned features of the previous layer.


This is the key difference comparing to othe ML algorithms which the model is limited to the features that we provide to it. As we discussed in [feautre engineering](feature_engineering.md), we can engineer new features from the existing features, but this is a manual process and in many cases we can't engineer all possible relevant features. Also, we could see that engineering new features can cause problems like [curse of dimensionality](feature_engineering.md#dimensionality-reduction) and [overfitting](generalization.md#overfitting). But in neural networks, the model can learn new features automatically and much more efficiently than manual feature engineering.

**Last Hidden Layer and Output Layer**:<br>
An intuitive way to think about a neural network is that just look at the last hidden layer and the output layer. This about the last hidden layer output as the input feature and the output layer as the model that predicts the target variable. But the key difference is that the last hidden layer features are **not** the original input features from our training dataset, but they are the **learned** and more complex features that learned from previous layers (other learned features) until they reach the last hidden layer. So, we changing the original input features with these learned features, to make the model more capable to learn complex patterns from the original data.

**Example**<br>
Let's say we want to classify images of persons. An image is a matrix of pixels. For example for a $100 \times 100$ pixel image, we have $10,000$ pixels. If the image is grayscale (black and white), each pixel has only one value between 0 and 255 which called _pixel intensity_ (a value between 0 which is the complete black and 255 which is the complete white). So, this image can be represented as a $100\times100$ matrix of scalar values between 0 and 255. We can represent this matrix as a vector of $10,000$ values.


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

Where:
- $x_i$ is the intensity of the $i^{th}$ pixel. $x_1$ is the first pixel in the first row, $x_2$ is the second pixel in the first row, $x_{101}$ is the first pixel in the second row, and so on.


> Most images use the RGB (Red, Green, Blue) color model, where the color of each pixel is determined by the combination of three color channels: red, green, and blue. Each channel has a value ranging from 0 to 255, where 0 represents the lowest intensity and 255 the highest intensity. So, if in our example the $100 \times 100$ image was colored, we would have 3 channels for each pixel (3 numbers for each pixel), and the image would be represented as a $100 \times 100 \times 3$ matrix of scalar values between 0 and 255, which can also be represented as a vector of $30,000$ values.

**Each layer learns a more complex feature from learned-features of the previous layer**<br>
The first layer of the neural network can learn features like edges, corners, and textures from the input image. The second layer can learn features like shapes, objects, and patterns from the learned features of the first layer. The third layer can learn features like faces, objects, and scenes from the learned features of the second layer. And so on.

![](images/nn_layers_learning_from_previous_layer.png)


Source: [Convolutional Deep Belief Networks
for Scalable Unsupervised Learning of Hierarchical Representations](https://web.eecs.umich.edu/~honglak/icml09-ConvolutionalDeepBeliefNetworks.pdf)


So, the neural network can learn complex patterns from the input data by learning new features from the learned features of the previous layer on its own. We don't need to engineer these features manually or even know what these features are. The neural network learns these features from low level features (like edges, corners, and textures) to high level features (like faces, objects, and scenes) completely on its own.


Further reading here: [Understanding Neural Networks Through Deep Visualization](https://arxiv.org/abs/1506.06579)

## Inference (Forward Pass)
Forward pass (also called **Forward Propagation**) is the process of passing the input features $\vec{\mathbf{x}}$ through the first layer, then calculate the output of the first layer $\vec{\mathbf{a}}^{[1]}$, then pass this output to the second layer and calculate the output of the second layer $\vec{\mathbf{a}}^{[2]}$, and so on until we reach the output layer.

As the name suggests, in the forward pass, we _pass_ the input features _forward_ through the network layer by layer to reach the output layer. In other words, we _propagate_ the activations of neurons from the input layer to the output layer.

**Each Layer Inputs a Vector and Outputs another Vector**:<br>
As we discussed, each layer inputs a vector of scalar values (features) and outputs another vector of scalar values (activation values of the neurons in that layer). The output of the layer is the input of the next layer and so on.

So, $\vec{\mathbf{a}}^{[1]}$ is the input vector (features) for the second layer, $\vec{\mathbf{a}}^{[2]}$ is the input vector for the third layer, and so on.

We can also denote input layer features $\vec{\mathbf{x}}$ as $\vec{\mathbf{a}}^{[0]}$ which represents the the output of the layer zero (input layer) and input vector of the layer 1.

> By convention, when we say a neural network has $l$ layers, it means it has $l-1$ hidden layers and 1 output layer. So, we count the output layer as a layer, but we don't count the input layer. The above example has 3 layers, 2 hidden layers and 1 output layer.

The following shows the details of the input and output of each layer and how they are calculated and connected to each other.

![](images/nn_forward_pass.png)

**Weights and Biases of Neurons**:<br>
Each neuron has a weight for each input feature and a bias term. So, if the input vector for a layer has $n$ features, each neuron in that layer has $n$ weights and 1 bias term. We denote them as $\vec{\mathbf{w}}^{[layer]}_{neuron}$ and $b^{[layer]}_{neuron}$. The weight vector is a vector of weights for each input feature to the neuron. For example, $\vec{\mathbf{w}}^{[2]}_{1}$ and $b^{[2]}_{1}$ are the weight vector and bias term for the first neuron in the second layer.

**Logit of a Neuron**:<br>
Logit of a neuron is calculated by dot product of the weight vector for that neuron and the input vector of the layer (output of the previous layer) plus the bias term:

$$z^{[l]}_{j} = \vec{\mathbf{w}}^{[l]}_{j} \cdot \vec{\mathbf{a}}^{[l-1]} + b^{[l]}_{j}$$

Where:
- $l$ is the layer number.
- $j$ is the neuron number in the layer.
- $\vec{\mathbf{w}}^{[l]}_{j}$ is the weight vector for the neuron $j$ in the layer $l$.
- $\vec{\mathbf{a}}^{[l-1]}$ is the output vector of the previous layer (input vector of the current layer).

So, the activation value for neuron for layer $l$ neuron $j$ is calculated as:

$$a^{[l]}_{j} = f(z^{[l]}_{j}) = f(\vec{\mathbf{w}}^{[l]}_{j} \cdot \vec{\mathbf{a}}^{[l-1]} + b^{[l]}_{j})$$


The output of the layer is calculated as by applying the activation function to the logit of each neuron in the layer:

$$\vec{\mathbf{a}}^{[l]} = f(\vec{\mathbf{z}}^{[l]})$$


**Activation Values are Scalar Numbers**<br>
Remember that these input and output vectors are the vectors of activation values which are scalar numbers. For example using imaginary numbers, the input and output of each layer can be represented as:

![](images/nn_vector_input_output.png)


- $\vec{\mathbf{x}}=\vec{\mathbf{a}}^{[0]}$ is a vector of 2 numbers which is the input vector for the first layer.
- $\vec{\mathbf{a}}^{[1]}$ is the output of the first layer which is a vector of 3 numbers. This vector is the input vector for the second layer.
- $\vec{\mathbf{a}}^{[2]}$ is the output of the second layer which is a vector of 2 numbers. This vector is the input vector for the third layer.
- $\vec{\mathbf{a}}^{[3]}$ is the output of the third layer (output layer) which is a vector of 1 number. This number is the output of the neural network and based on that we can predict the target variable $\hat{y}$.

**Activation Functions for Each Layer**:<br>
Each layer can have a different activation function.
- The hidden layers can have the same activation function or different activation functions depending on the problem and performance.
- The **output layer** is determined by the type of the problem we are solving. The output of this layer should provide the model's prediction, so we choose the activation function of this layer based on what we are predicting. For example, if we are solving a binary classification problem, we can use the Sigmoid Function as the activation function of the output layer, and if we are solving a multi-class classification problem, we can use the **Softmax** Function as the activation function of the output layer.

**Derive the Prediction of $\hat{y}$ from the Output of the Neural Network**:<br>
The output of the neural network is the output of the output layer. The output layer is the last layer of the neural network.

Let's say in the above example we are solving a binary classification problem, and we designed the output layer to have only one neuron with Sigmoid function as the activation function. The output of the neural network is the output of this neuron:

$$\vec{\mathbf{a}}^{[3]} = \begin{bmatrix} a^{[3]}_1 \\
\end{bmatrix}$$
Which

$$a^{[3]}_1 = \sigma(z^{[3]}_1) = \sigma(\vec{\mathbf{w}}^{[3]}_{1} \cdot \vec{\mathbf{a}}^{[2]} + b^{[3]}_{1})$$

If we set the threshold of the Sigmoid function to 0.5, the output of the neural network is:

$$\hat{y} = 1 \quad \text{if} \quad a^{[3]}_1 \geq 0.5$$
$$\hat{y} = 0 \quad \text{if} \quad a^{[3]}_1 < 0.5$$


So, in the above example the model's prediction is $1$.

$$a^{[3]}_1=0.72 \Rightarrow \hat{y}=1$$
