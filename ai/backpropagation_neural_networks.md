---
date: "2025-03-16"
draft: false
title: "Backpropagation in Neural Networks"
description: "A deep dive into the backpropagation algorithm, computational graph, and how it works in neural networks."
tags:
    - "AI"
    - "Neural Networks"
---

In the following we'll use an example to illustrate how backpropagation in neural networks works in detail. The example we try to solve a simple binary classification problem using a simple feedforward neural network consisting of fully connected (dense) layers.

This [backpropagation illustration](https://developers.google.com/machine-learning/crash-course/backprop-scroll) by Google is a good visualization of the process.

In this example we go through the forward propagation of a simple neural network and then step by step details of the backpropagation using computational graph and chain rule.

Let's say we have the following neural network which is used for binary classification. We have the following:

**Input**:<br>
2 samples with 2 features

**Network Architecture**:<br>
- Layer 1: Fully connected layer with 3 neurons and ReLU activation function.
- Layer 2: Fully connected layer with 2 neurons and ReLU activation function.
- Layer 3 (output): Fully connected layer with 1 neuron (output) and Sigmoid activation function.

**Loss function**:<br>
Binary Cross Entropy

![](https://pooya.io/ai/images/nn_backpropagation.svg)




In this example we use a batch dataset with 2 samples. The input $X$ and target $Y$ are defined as follows:

$$X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$
$$\vec{\mathbf{y}} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

Which means, for example 1 $x_1 = 1$ and $x_2 = 2$ and the target class $y = 0$.

Recall that we maintain each sample in **rows** and features in **columns**. So, each row of $X$ and $\vec{\mathbf{y}}$ is associated with one sample.


```python
import torch

X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[0.0], [1.0]])
```

## Define the Neural Network

Let's create our neural network


```python
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        # Define the model architecture (Layers and nodes)
        self.linear1 = nn.Linear(in_features=2, out_features=3)
        self.linear2 = nn.Linear(in_features=3, out_features=2)
        self.linear3 = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        # Forward Propagation happens here.
        # It takes the input tensor x and returns the output tensor for each
        # layer by applying the linear transformation first and then the
        # activation function.
        # It start from layer 1 and goes forward layer by layer to the output
        # layer.

        # Layer 1 linear transformation
        Z1 = self.linear1(x)
        # Layer 1 activation
        A1 = F.relu(Z1)

        # Layer 2 linear transformation
        Z2 = self.linear2(A1)
        # Layer 2 activation
        A2 = F.relu(Z2)

        # Layer 3 (output layer) linear transformation
        Z3 = self.linear3(A2)
        # Layer 3 activation
        A3 = F.sigmoid(Z3)

        # Output of the model A3, along with the intermediate results
        return A3, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3}
```

In practice, for classification problems, when we use the Sigmoid or Softmax activation function in the output layer, we defer the activation of the output layer to the outside the model. In other words, the output layer just do the linear transformation $Z$ and output the [logits](https://pooya.io/ai/forward-propagation-neural-networks/). Then we apply the activatio function (Sigmoid or Softmax) outside the model on the logits to get the predicted probabilities.

This approach is the same for both inference and training.

However, in this example, for simplicity and focus on the backpropagation, we will include the Sigmoid activation function in the output layer. So, in this example, the output layer will output the predicted probabilities.


```python
model = NeuralNet()
print(model)
```

    NeuralNet(
      (linear1): Linear(in_features=2, out_features=3, bias=True)
      (linear2): Linear(in_features=3, out_features=2, bias=True)
      (linear3): Linear(in_features=2, out_features=1, bias=True)
    )


Let's see the initial weights and biases of our neural network.


```python
def print_model_parameters(model):
    for i, child in enumerate(model.children()):
        print(f"Layer {i+1}: {type(child).__name__}")
        child_parameters = dict(child.named_parameters())

        for name, param in child_parameters.items():
            print(f"\n{name}: {param.size()} {param}")
            print(f"{name}.grad:\n{param.grad}")

        print("-" * 50)


print_model_parameters(model)
```

    Layer 1: Linear

    weight: torch.Size([3, 2]) Parameter containing:
    tensor([[ 0.6693,  0.6654],
            [-0.6775, -0.2974],
            [-0.3911,  0.6824]], requires_grad=True)
    weight.grad:
    None

    bias: torch.Size([3]) Parameter containing:
    tensor([-0.4274,  0.4013, -0.0109], requires_grad=True)
    bias.grad:
    None
    --------------------------------------------------
    Layer 2: Linear

    weight: torch.Size([2, 3]) Parameter containing:
    tensor([[-0.4749, -0.2138, -0.0369],
            [-0.0278,  0.4659, -0.1942]], requires_grad=True)
    weight.grad:
    None

    bias: torch.Size([2]) Parameter containing:
    tensor([-0.4565, -0.3265], requires_grad=True)
    bias.grad:
    None
    --------------------------------------------------
    Layer 3: Linear

    weight: torch.Size([1, 2]) Parameter containing:
    tensor([[-0.2296,  0.4444]], requires_grad=True)
    weight.grad:
    None

    bias: torch.Size([1]) Parameter containing:
    tensor([0.6471], requires_grad=True)
    bias.grad:
    None
    --------------------------------------------------


As we expect, gradients of parameters are `None` since we haven't computed any gradients yet.

For this example for simplicity and having reproducible results, we'll set the weights and biases manually. Let's say we have the following weights and biases:

Similar to the way that PyTorch creates weights matrices:
- Each row of $W^{[l]}$ is associated with one neuron in the layer $l$. For example, in layer 1, We have 3 neurons, so we have 3 rows in $W^{[1]}$.
- Each column of $W^{[l]}$ is associated with one feature of input values. For example, the number of columns in $W^{[1]}$ is equal to the number of features in the input layer $X$. We have 2 features in the input layer $X$, so we have 2 columns in $W^{[1]}$.


**Layer 1 (3 neurons):**
$$W^{[1]} = \begin{bmatrix} -1 & 2 \\ 3 & 0.5 \\ -0.1 & -4\end{bmatrix} \quad {\vec{\mathbf{b}}}^{[1]} = \begin{bmatrix} 1 & -2 & 0.3 \end{bmatrix}$$


**Layer 2 (2 neurons):**
$$W^{[2]} = \begin{bmatrix} 0.5 & 1 & -2 \\ 0.7 & 0.1 & 0.3\end{bmatrix} \quad {\vec{\mathbf{b}}}^{[2]} = \begin{bmatrix} -4 & 5 \end{bmatrix}$$

**Layer 3 (output):**
$$W^{[3]} = \begin{bmatrix} 0.5 & -0.3 \end{bmatrix} \quad {\vec{\mathbf{b}}}^{[3]} = \begin{bmatrix} 0.1 \end{bmatrix}$$

Note: The number of weight and biases are independent of the number of training samples (in any batch or entire dataset). The whole point of training with sample datasets is to optimize these parameters by exposing them to the entire dataset through cycle of forward and backward propagation. So, no matter what is the size of the dataset, the number of parameters in the model is fixed and defined by the architecture of the neural network.



```python
# Layer 1
W1 = torch.tensor([[-1.0, 2.0], [3.0, 0.5], [-0.1, -4.0]], requires_grad=True)
b1 = torch.tensor([1.0, -2.0, 0.3], requires_grad=True)

# Layer 2
W2 = torch.tensor([[0.5, 1.0, -2.0], [0.7, 0.1, 0.3]], requires_grad=True)
b2 = torch.tensor([-4.0, 5.0], requires_grad=True)

# Layer 3 (Output layer)
W3 = torch.tensor([[0.5, -0.3]], requires_grad=True)
b3 = torch.tensor([0.1], requires_grad=True)
```

Now we set these weights and biases in our model.


```python
model.linear1.weight.data.copy_(W1)
model.linear1.bias.data.copy_(b1)

model.linear2.weight.data.copy_(W2)
model.linear2.bias.data.copy_(b2)

model.linear3.weight.data.copy_(W3)
model.linear3.bias.data.copy_(b3)

print_model_parameters(model)
```

    Layer 1: Linear

    weight: torch.Size([3, 2]) Parameter containing:
    tensor([[-1.0000,  2.0000],
            [ 3.0000,  0.5000],
            [-0.1000, -4.0000]], requires_grad=True)
    weight.grad:
    None

    bias: torch.Size([3]) Parameter containing:
    tensor([ 1.0000, -2.0000,  0.3000], requires_grad=True)
    bias.grad:
    None
    --------------------------------------------------
    Layer 2: Linear

    weight: torch.Size([2, 3]) Parameter containing:
    tensor([[ 0.5000,  1.0000, -2.0000],
            [ 0.7000,  0.1000,  0.3000]], requires_grad=True)
    weight.grad:
    None

    bias: torch.Size([2]) Parameter containing:
    tensor([-4.,  5.], requires_grad=True)
    bias.grad:
    None
    --------------------------------------------------
    Layer 3: Linear

    weight: torch.Size([1, 2]) Parameter containing:
    tensor([[ 0.5000, -0.3000]], requires_grad=True)
    weight.grad:
    None

    bias: torch.Size([1]) Parameter containing:
    tensor([0.1000], requires_grad=True)
    bias.grad:
    None
    --------------------------------------------------


## Step 1: Forward Propagation

Now let's run the forward propagation using the current weights and biases, and the input $X$.


```python
# Forward Propagation
output, model_results = model(X)

# Print the intermediate results
print(
    "Intermediate results:\n"
    f"Z1:\n{model_results["Z1"]}\n"
    f"A1:\n{model_results["A1"]}\n"
    f"Z2:\n{model_results["Z2"]}\n"
    f"A2:\n{model_results["A2"]}\n"
    f"Z3:\n{model_results["Z3"]}\n"
    f"A3 (Model Output):\n{output}"
)
```

    Intermediate results:
    Z1:
    tensor([[  4.0000,   2.0000,  -7.8000],
            [  6.0000,   9.0000, -16.0000]], grad_fn=<AddmmBackward0>)
    A1:
    tensor([[4., 2., 0.],
            [6., 9., 0.]], grad_fn=<ReluBackward0>)
    Z2:
    tensor([[ 0.0000,  8.0000],
            [ 8.0000, 10.1000]], grad_fn=<AddmmBackward0>)
    A2:
    tensor([[ 0.0000,  8.0000],
            [ 8.0000, 10.1000]], grad_fn=<ReluBackward0>)
    Z3:
    tensor([[-2.3000],
            [ 1.0700]], grad_fn=<AddmmBackward0>)
    A3 (Model Output):
    tensor([[0.0911],
            [0.7446]], grad_fn=<SigmoidBackward0>)


We'll follow the steps manually to understand the computational graph and forward propagation.

![](https://pooya.io/ai/images/nn_computational_graph.svg)

As it shown in the above graph, we start from the first node and go through the graph from left to right.

In [Forward Propagation](https://pooya.io/ai/forward-propagation-neural-networks/) we feed the input $X$ (which could be a single sample, a batch of samples, or the entire dataset) to the model and then compute the output of first layer, then give that output to the next layer (as input) and compute the output of the next layer, and so on until we reach the output layer.

In each layer, we have two steps of computation:

**1. Linear Transformation:**<br>
$$Z^{[l]} = A^{[l-1]} \cdot {W^{[l]}}^\top + {\vec{\mathbf{b}}}^{[l]}$$

**2. Activation Function:**<br>
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

By convention, we consider $X$ as the layer $0$. So, $A^{[0]} = X$.

Let's calculate the output of the layer $1$.

**Layer 1:**
$$Z^{[1]} = X \cdot {W^{[1]}}^\top + {\vec{\mathbf{b}}}^{[1]}$$

$$Z^{[1]} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \cdot \begin{bmatrix} -1 & 3 & -0.1 \\ 2 & 0.5 & -4 \end{bmatrix} + \begin{bmatrix} 1 & -2 & 0.3 \end{bmatrix}$$


$$Z^{[1]} = \begin{bmatrix} 3 & 4 & -8.1 \\ 5 & 11 & -16.3 \end{bmatrix} + \begin{bmatrix} 1 & -2 & 0.3 \end{bmatrix}$$

We broadcast the bias vector to the shape of $(2, 3)$ and add it to the dot product of $X$ and $W^{[1]}$.

$$Z^{[1]} = \begin{bmatrix} 3 & 4 & -8.1 \\ 5 & 11 & -16.3 \end{bmatrix} + \begin{bmatrix} 1 & -2 & 0.3 \\ 1 & -2 & 0.3 \end{bmatrix} = \begin{bmatrix} 4 & 2 & -7.8 \\ 6 & 9 & -16 \end{bmatrix}$$

Let's verify our calculation with PyTorch.


```python
print(f"Z1:\n{model_results["Z1"]}")
```

    Z1:
    tensor([[  4.0000,   2.0000,  -7.8000],
            [  6.0000,   9.0000, -16.0000]], grad_fn=<AddmmBackward0>)


Now let's calculate the activation of layer 1 using the ReLU activation function.

$$A^{[1]} = \text{ReLU}(Z^{[1]})$$

$$A^{[1]} = \begin{bmatrix} \text{ReLU}(4) & \text{ReLU}(2) & \text{ReLU}(-7.8) \\ \text{ReLU}(6) & \text{ReLU}(9) & \text{ReLU}(-16) \end{bmatrix}$$

We know that the ReLU function is defined as:
$$\text{ReLU}(z) = \max(0, z)$$

We apply ReLU element-wise to the matrix $Z^{[1]}$. So, the output of the layer 1 is:

$$A^{[1]} = \begin{bmatrix} 4 & 2 & 0 \\ 6 & 9 & 0 \end{bmatrix}$$

Let's verify our calculation with PyTorch.


```python
print(f"A1:\n{model_results["A1"]}")
```

    A1:
    tensor([[4., 2., 0.],
            [6., 9., 0.]], grad_fn=<ReluBackward0>)


Now if we compare this result with the PyTorch output, we see that they are the same.


Now let's calculate the output of the layer 2.

**Layer 2:**

$$Z^{[2]} = A^{[1]} \cdot {W^{[2]}}^\top + {\vec{\mathbf{b}}}^{[2]}$$

$$Z^{[2]} = \begin{bmatrix} 4 & 2 & 0 \\ 6 & 9 & 0 \end{bmatrix} \cdot \begin{bmatrix} 0.5 & 0.7 \\ 1 & 0.1 \\ -2 & 0.3 \end{bmatrix} + \begin{bmatrix} -4 & 5 \end{bmatrix}$$

Which equals to:

$$Z^{[2]} = \begin{bmatrix} 0 & 8 \\ 8 & 10.1 \end{bmatrix}$$

And then applying the ReLU activation function:

$$A^{[2]} = \begin{bmatrix} 0 & 8 \\ 8 & 10.1 \end{bmatrix}$$


```python
print(f"Z2:\n{model_results["Z2"]}")
print(f"A2:\n{model_results["A2"]}")
```

    Z2:
    tensor([[ 0.0000,  8.0000],
            [ 8.0000, 10.1000]], grad_fn=<AddmmBackward0>)
    A2:
    tensor([[ 0.0000,  8.0000],
            [ 8.0000, 10.1000]], grad_fn=<ReluBackward0>)


In the same way, we can keep going **forward** and compute the outputs (linear transformations and activations) layer by layer until we reach the output layer.

The output of the output layer is the **prediction** of the model which in this case is the predicted probability of binary classification.

## Step 2: Compute the Loss and Cost

Computing the cost will provide the error of our model in respect to the labels (target value $Y$). To calculate the loss function, we continue moving forward (left to right) in the computational graph.

The [cost](https://pooya.io/ai/loss-cost-functions-machine-learning/) function is usually the average of the **loss** function over all the samples in the batch (which pass through in the forward propagation).



The loss function for binary classification is the [Binary Cross-Entropy (BCE)](https://pooya.io/ai/loss-cost-functions-machine-learning/) loss which is defined as:
$$
L(f_{\theta}(x^{(i)}), y^{(i)}) = \begin{cases}
    - \log(f_{\theta}(x^{(i)})) & \text{if $y^{(i)}=1$}\\
    - \log(1 - f_{\theta}(x^{(i)})) & \text{if $y^{(i)}=0$}
  \end{cases}
$$

Where:
- $i$ is the index of the sample in the batch.
- $y$ is the target value (label) of the $i$-th sample.
- $\theta$ is the model's parameters (weights and biases).
- $f_{\theta}$ is the model's function which produces the predicted probability based on the input $x$ and the model's parameters $\theta$.


We have the following target values for our samples:
$$\vec{\mathbf{y}} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

$A^{[3]}$ is the output of the model (layer 3) which is the predicted probability of the model. Each row of $A^{[3]}$ is the predicted probability of the corresponding sample. For example, $a_{1}^{[3]}$ is the predicted probability of the first sample, and $a_{2}^{[3]}$ is the predicted probability of the second sample.

$$A=\begin{bmatrix}
    a_{1}^{[3]} \\
    a_{2}^{[3]}
\end{bmatrix} = \begin{bmatrix}
    0.0911 \\
    0.7446
\end{bmatrix}$$


$y$ for the first sample is 0, and for the second sample is 1. So, the loss function is a matrix with 2 rows and 1 column. The first row is the loss for the first sample, and the second row is the loss for the second sample.

$$L=\begin{bmatrix}
    -\log(1 - {a_{1}}^{[3]}) \\
    -\log({a_{2}}^{[3]})
\end{bmatrix} = \begin{bmatrix}
    -\log(1 - 0.0911) \\
    -\log(0.7446)
\end{bmatrix} = \begin{bmatrix}
    0.0955 \\
    0.2949
\end{bmatrix}$$


The cost function is the average of the loss function over all the samples in the batch.

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} L(\theta)^{(i)}$$

Where:
- $m$ is the number of samples in the batch.

For more details see [Loss and Cost Functions in Machine Learning](https://pooya.io/ai/loss-cost-functions-machine-learning/).


So, the cost function is:

$$J(\theta) = \frac{1}{2} \left( -\log(1 - {a_{1}}^{[3]}) -\log({a_{2}}^{[3]}) \right)$$

Which if we plug in the values of $a_{1}^{[3]}$ and $a_{2}^{[3]}$ we get:

$$J(\theta) = \frac{1}{2} \left( -\log(1 - 0.0911) -\log(0.7446) \right) = 0.1952$$

Now, let's use  pytorch builtin cost function to calculate the cost.


```python
cost = F.binary_cross_entropy(output, y)

print(f"Cost: {cost}")
```

    Cost: 0.19522885978221893


For more stable computation, in practice, we usually don't include the activation function in the output layer. So, the output of the model is the linear transformation $Z$ of the output layer (logits).

In this example, for simplicity, we include the Sigmoid activation function in the output layer. So, the `output` is the predicted probabilities. We will use `binary_cross_entropy()` loss function to calculate the loss. If we had deferred the activation function to the outside of the model, then the output would be the logits of the output layer, which then we should have used `binary_cross_entropy_with_logits()` loss function instead.


## Step 3: Backpropagation

So far, we calculate the output (inference) of the model and the cost (error) of the model in comparison with the target values. Now we need to start optimizing the model's parameters by minimizing the error (cost). For doing this we need to calculate the gradients of the cost function with respect to the model's parameters (weights and biases) and then update the parameters using the gradients.


The backpropagation algorithm is a method for calculating the gradients of the cost function with respect to the model's parameters (weights and biases) using the chain rule of calculus and the computational graph of the neural network.

In backprop, we calculate the gradients of the loss function with respect to each parameter of the model. As we discussed in the [Computational Graph](https://pooya.io/ai/computational-graph-machine-learning/), we start from the last node of the computational graph (the cost node) and then calculate the partial derivative (gradient) of the loss with respect to each part of the graph step by step in backward direction until we reach to all the parameters of the model. Hence, the name **backpropagation** or **backward pass**.

**Right-to-Left**<br>
![](https://pooya.io/ai/images/nn_computational_graph.svg)


**Using Chain Rule:**<br>
We can see the whole model as a huge composite function which is made of many smaller functions (linear transformation and activation function of each layer). These functions are composed together layer by layer like a chain. So, in simple terms we can say that we use chain rule to calculate the gradient of the loss with respect to each parameter from the most outer function (cost) to the most inner function (parameters of the model).

**Gradient of Loss vs Cost**:<br>
We calculate the partial derivative of the **loss** with respect to each parameter of the model. That gives us the gradient of the loss with respect to each parameter for **one single sample**. Then we calculate the average of these gradients (mean gradient) over all the samples in the batch. In that case, we can say we have calculated the gradient of the **cost** with respect to each parameter of the model.

Let's start the backward propagation by first defining our optimizer. Optimizer is the overall optimization algorithm which calculate the gradients and then update the parameters of the model. We define the simple [Stochastic Gradient Descent (SGD)](https://pooya.io/ai/gradient-descent-machine-learning/) algorithm as our optimizer. However, this example, mainly focus on the backpropagation algorithm, stepping backward through the computational graph and calculating the gradients.


```python
import torch.optim as optim

# Define Stochastic Gradient Descent (SGD) with learning rate of 0.01
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

Let's set all the gradients to zero before starting any new computation. PyTorch internally stores the gradients of the parameters in the `grad` attribute of the parameters. So, the `zero_grad()` method of the optimizer sets all the gradients to zero. It may seem unnecessary to reset the gradients at this point (since we haven't calculated any gradients yet), but as a good practice before starting the backpropagation, we reset the gradients of all the parameters.

In this particular example, as we haven't calculated any gradients yet, the gradients are `None`. So, resetting them has no effect.


```python
optimizer.zero_grad()

print_model_parameters(model)
```

    Layer 1: Linear

    weight: torch.Size([3, 2]) Parameter containing:
    tensor([[-1.0000,  2.0000],
            [ 3.0000,  0.5000],
            [-0.1000, -4.0000]], requires_grad=True)
    weight.grad:
    None

    bias: torch.Size([3]) Parameter containing:
    tensor([ 1.0000, -2.0000,  0.3000], requires_grad=True)
    bias.grad:
    None
    --------------------------------------------------
    Layer 2: Linear

    weight: torch.Size([2, 3]) Parameter containing:
    tensor([[ 0.5000,  1.0000, -2.0000],
            [ 0.7000,  0.1000,  0.3000]], requires_grad=True)
    weight.grad:
    None

    bias: torch.Size([2]) Parameter containing:
    tensor([-4.,  5.], requires_grad=True)
    bias.grad:
    None
    --------------------------------------------------
    Layer 3: Linear

    weight: torch.Size([1, 2]) Parameter containing:
    tensor([[ 0.5000, -0.3000]], requires_grad=True)
    weight.grad:
    None

    bias: torch.Size([1]) Parameter containing:
    tensor([0.1000], requires_grad=True)
    bias.grad:
    None
    --------------------------------------------------


We can see that each parameter (which has `requires_grad=True`) has a `grad` attribute which stores the gradient of the loss with respect to that parameter. We can see that all of our parameters currently has `None` as their gradients.

Now let's run the backpropagation by calling the `backward()` method on the last node of the computational graph (the cost node). This will start the backward step by step calculation of the gradients of the cost with respect to each parameter of the model.


```python
# Backpropagation (compute the gradients)
cost.backward()
```

Now we can see that the gradients of the parameters are calculated and stored in the `grad` attribute of each parameter.


```python
print_model_parameters(model)
```

    Layer 1: Linear

    weight: torch.Size([3, 2]) Parameter containing:
    tensor([[-1.0000,  2.0000],
            [ 3.0000,  0.5000],
            [-0.1000, -4.0000]], requires_grad=True)
    weight.grad:
    tensor([[-0.0249, -0.0396],
            [-0.1814, -0.2428],
            [ 0.0000,  0.0000]])

    bias: torch.Size([3]) Parameter containing:
    tensor([ 1.0000, -2.0000,  0.3000], requires_grad=True)
    bias.grad:
    tensor([-0.0147, -0.0614,  0.0000])
    --------------------------------------------------
    Layer 2: Linear

    weight: torch.Size([2, 3]) Parameter containing:
    tensor([[ 0.5000,  1.0000, -2.0000],
            [ 0.7000,  0.1000,  0.3000]], requires_grad=True)
    weight.grad:
    tensor([[-0.3831, -0.5747,  0.0000],
            [ 0.1752,  0.3175,  0.0000]])

    bias: torch.Size([2]) Parameter containing:
    tensor([-4.,  5.], requires_grad=True)
    bias.grad:
    tensor([-0.0639,  0.0246])
    --------------------------------------------------
    Layer 3: Linear

    weight: torch.Size([1, 2]) Parameter containing:
    tensor([[ 0.5000, -0.3000]], requires_grad=True)
    weight.grad:
    tensor([[-1.0216, -0.9253]])

    bias: torch.Size([1]) Parameter containing:
    tensor([0.1000], requires_grad=True)
    bias.grad:
    tensor([-0.0821])
    --------------------------------------------------


Now, let's go through the steps of the backpropagation manually.

Remember we start with the final node and walk backward the computational graph. So, we start with the cost node and calculate the gradient of the cost with respect to the output of the output layer.

> Important note is that all of what we calculate in forward and backward propagations is done using **matrix** operations.

Also, recall the word **partial derivative** and **gradient** mean the same thing and we use them interchangeably.

As we discussed earlier, we calculate the gradient of **loss** with respect to each parameter of the model. We do this for all of the calculations. At the end, we calculate the average of these gradients (mean gradient) over all the samples in the batch to give the gradient of the **cost** with respect to each parameter of the model.

So, all the following steps are computing the gradient of the **loss** with respect to each parameter of the model.

### 1. Gradient of $J$ with respect to $L$
We start our way from the the last node of the computational graph which is the cost node. This is a scalar-valued function $J(\theta)$ (cost function). The cost function is a scalar value which is (commonly) the average of the loss function over all the samples in the batch.

As we discussed earlier, the cost function is defined as:
$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} L(\theta)^{(i)}$$

Which in this case:

$$J(\theta) = \frac{1}{2} L(\theta)^{(1)} + \frac{1}{2} L(\theta)^{(2)}$$

Where $l(\theta)^{(i)}$ is the loss function for the $i$-th sample.


We want to calculate the gradient of the cost function with respect to the loss function. The cost function is a scalar value, but the loss is a matrix $(2 \times 1)$. So, we need to use derivative for [Scalar-valued Function of a Matrix](https://pooya.io/math/derivative-matrices/)

So, the derivative of a scalar-valued function $J$ with respect to a matrix $L$ is:

$$\frac{\partial J}{\partial L} = \begin{bmatrix} \frac{\partial J}{\partial {l_{1}}^{(1)}} \\ \frac{\partial J}{\partial {l_{2}^{(2)}}} \end{bmatrix}=\begin{bmatrix} \frac{1}{2} \\ \frac{1}{2} \end{bmatrix}$$

The loss matrix $l$ always a column vecotr with only 1 column and $m$ rows, where $m$ is the number of samples in the batch, and each element in that row is the scalar value of the loss of the corresponding sample.


### 2. Gradient of $J$ with respect to $A^{[3]}$
Now we go one step back in the computational graph and calculate the gradient of the cost with respect to the output of the output layer $A^{[3]}$. Using the chain rule:

$$\frac{\partial J(\theta)}{\partial {A^{[3]}}}=\frac{\partial J(\theta)}{\partial L(\theta)} \cdot \frac{\partial L(\theta)}{\partial {A^{[3]}}}$$

We have already calculated the $\frac{\partial J(\theta)}{\partial L(\theta)}$ in the previous step. Now we need to calculate the $\frac{\partial L(\theta)}{\partial {A^{[3]}}}$.

Both $L$ and $A^{[3]}$ are matrices. So, we need to use derivative for [Vector-valued Function of a Matrix](https://pooya.io/math/derivative-matrices/). So, we need to first vectorize (flatten) both $L$ and $A^{[3]}$ matrices to vecotrs. Then create a **Jacobian matrix** for the function $L$ with respect to $A^{[3]}$. The Jacobian matrix is a matrix of all first-order partial derivatives of the function.

**Jacobian Matrix of $L$ with respect to $A^{[3]}$**:<br>
We have the following matrices $L$ with the shape of $(2 \times 1)$ and $A^{[3]}$ with the shape of $(2 \times 1)$.

$$L = {\begin{bmatrix} {l_{1}}^{(1)} \\ {l_{2}}^{(2)} \end{bmatrix}}_{2 \times 1} \quad A^{[3]} = {\begin{bmatrix} {a_{1}}^{[3]^{(1)}} \\ {a_{2}}^{[3]^{(2)}} \end{bmatrix}}_{2 \times 1}$$


Jacobian matrix of $L$ with respect to $A^{[3]}$ is shape of $(2 \times 1) \times (1 \times 2) = (2 \times 2)$.
> To be consistent with PyTorch (and other deep learning libraries), we use Row-major order for the vectorization. So, we flatten the matrix by stacking the rows of the matrix one after another.

$$\text{vec}(L) = \begin{bmatrix} {l_{1}}^{(1)} & {l_{2}}^{(2)} \end{bmatrix}$$
$$\text{vec}(A^{[3]}) = \begin{bmatrix} {a_{1}}^{[3]^{(1)}} & {a_{2}}^{[3]^{(2)}} \end{bmatrix}$$

So, partial derivative of $L$ with respect to $A^{[3]}$ is defined as a Jacobian matrix of $L$ with respect to $A^{[3]}$. The Jacobian matrix is partial derivative of all the elements of $L$ with respect to all the elements of $A^{[3]}$. So, we have the following Jacobian matrix:

$$
\frac{\partial\, \text{vec}(L)}{\partial \, \text{vec}(A^{[3]})} = \text{Jacobian}_{L, A^{[3]}} = \begin{bmatrix} \frac{\partial {l_{1}}^{(1)}}{\partial {a_{1}}^{[3]^{(1)}}} & \frac{\partial {l_{1}}^{(1)}}{\partial {a_{2}}^{[3]^{(2)}}} \\ \frac{\partial {l_{2}}^{(2)}}{\partial {a_{1}}^{[3]^{(1)}}} & \frac{\partial {l_{2}}^{(2)}}{\partial {a_{2}}^{[3]^{(2)}}} \end{bmatrix}$$



Let's calculate each element of the Jacobian matrix. But before that let's define the loss function $L$ for the binary classification. The loss function is defined as:

$$l^{(i)} = \begin{cases}
    -\log(a^{[3]^{(i)}}) & \text{if $y^{(i)}=1$}\\
    -\log(1 - a^{[3]^{(i)}}) & \text{if $y^{(i)}=0$}
  \end{cases}$$

So, partial derivatie of $l^{(i)}$ with respect to $a^{[3]^{(i)}}$ is:

$$\frac{\partial l^{(i)}}{\partial a^{[3]^{(i)}}} = \begin{cases}
    -\frac{1}{a^{[3]^{(i)}}} & \text{if $y^{(i)}=1$}\\
    \frac{1}{1 - a^{[3]^{(i)}}} & \text{if $y^{(i)}=0$}
  \end{cases}$$


So, now let's each element of the Jacobian matrix:

**Element 1,1**:<br>
This is the the partial derivative of the loss function of the _first example_ with respect to the output of the output layer of the _first example_. So, we have $y^{(1)}=0$ and ${a_{1}}^{[3]^{(1)}}=0.0911$.

$$
\frac{\partial {l_{1}}^{(1)}}{\partial {a_{1}}^{[3]^{(1)}}} = \frac{1}{1 - {a_{1}}^{[3]^{(1)}}} = \frac{1}{1 - 0.0911} = \frac{1}{0.9089}
$$

**Element 1,2**:<br>
This is the the partial derivative of the loss function of the _first example_ with respect to the output of the output layer of the _second example_. Output of the second example does not affect the loss of the first example. So, this element is $0$.

$$
\frac{\partial {l_{1}}^{(1)}}{\partial {a_{2}}^{[3]^{(2)}}} = 0
$$

**Element 2,1**:<br>
This is the the partial derivative of the loss function of the _second example_ with respect to the output of the output layer of the _first example_. Output of the first example does not affect the loss of the second example. So, this element is also $0$.

$$
\frac{\partial {l_2}^{(2)}}{\partial {a_1}^{[3]^{(1)}}} = 0
$$


**Element 2,2**:<br>
This is the the partial derivative of the loss function of the _second example_ with respect to the output of the output layer of the _second example_. So, we have $y^{(2)}=1$ and $a^{[3]^{(2)}}=0.7446$.

$$
\frac{\partial {l_2}^{(2)}}{\partial {a_2}^{[3]^{(2)}}} = -\frac{1}{{a_2}^{[3]^{(2)}}} = -\frac{1}{0.7446}
$$

So, the Jacobian matrix is:

$$
\frac{\partial L}{\partial A^{[3]}} = \begin{bmatrix} \frac{1}{0.9089} & 0 \\ 0 & -\frac{1}{0.7446} \end{bmatrix}
$$

Now if we get back to the chain rule we defined earlier:

$$\frac{\partial J(\theta)}{\partial {A^{[3]}}}=\frac{\partial J(\theta)}{\partial L(\theta)} \cdot \frac{\partial L(\theta)}{\partial {A^{[3]}}}$$

We need to also flatten the first matrix $\frac{\partial J(\theta)}{\partial L(\theta)}$ to a vector (with the same order as the Jacobian matrix). So, we have:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(L(\theta))} = \begin{bmatrix} \frac{1}{2} & \frac{1}{2} \end{bmatrix}
$$


So we can write the chain rule in vectorized form as:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(A^{[3]})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(L(\theta))} \cdot \frac{\partial \,\text{vec}(L(\theta))}{\partial \, \text{vec}(A^{[3]})}
$$

So, if we plug in the values we have:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(A^{[3]})} = \begin{bmatrix} \frac{1}{2} & \frac{1}{2} \end{bmatrix} \cdot \begin{bmatrix} \frac{1}{0.9089} & 0 \\ 0 & -\frac{1}{0.7446} \end{bmatrix}$$

Which is equal to:
$$ \begin{bmatrix} \frac{1}{2 \times 0.9089} & -\frac{1}{2 \times 0.7446} \end{bmatrix}
= \begin{bmatrix} 0.5501 & -0.6715 \end{bmatrix}
$$

Now in order to get the final result, we have to reverse the flattening operation (de-vectorize) in the exactly same way we flattened the matrix. So, we need to reshape this result to the shape of $A^{[3]}$ which is $(2 \times 1)$. So, we have:

$$
\frac{\partial J(\theta)}{\partial {A^{[3]}}} = \begin{bmatrix} 0.5501 \\ -0.6715 \end{bmatrix}
$$

**$X$ and $\vec{\mathbf{y}}$ are constants**:<br>
In Backpropagation, the goal is find the gradient of the loss with respect to the parameters of the mode. So, as per rule of partial derivative, all variables except the one that we are taking the derivative with respect to are considered as constants. So, here the input value $X$ and the target value $\vec{\mathbf{y}}$ are considered as constants in our computation.

### 3. Gradient of $J$ with respect to $Z^{[3]}$

We go one step back in the computational graph and calculate the gradient of the cost with respect to the linear transformation of the output layer.

So, using the chain rule, we can write:

$$\frac{\partial J(\theta)}{\partial {Z^{[3]}}} = \frac{\partial J(\theta)}{\partial {A^{[3]}}} \cdot \frac{\partial {A^{[3]}}}{\partial {Z^{[3]}}}$$

We already calculated $\frac{\partial J(\theta)}{\partial {A^{[3]}}}$ in the previous step. So, we need to calculate $\frac{\partial {A^{[3]}}}{\partial {Z^{[3]}}}$.



$A^{[3]}$ is a function of $Z^{[3]}$ through the Sigmoid activation function.

$$A^{[3]} = \sigma(Z^{[3]})$$


Where $\sigma$ is the Sigmoid activation function. For the Sigmoid function $\sigma(x)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

The [Derivative of Sigmoid function](https://pooya.io/math/derivatives/#derivative-of-sigmoid-function) is as follows:

$$\frac{d\sigma(x)}{dx} = \sigma(x) \cdot (1 - \sigma(x))$$


In our case, $A^{[3]}$ and $Z^{[3]}$ are both matrices. So, we need to use derivative for Vector-valued Function of a Matrix, which means we need to use Jacobian matrix to calculate the derivative of $A^{[3]}$ with respect to $Z^{[3]}$ similar to the previous step.

We have the following matrices:

$$A^{[3]} = {\begin{bmatrix} {a_1}^{[3]^{(1)}} \\ {a_2}^{[3]^{(2)}} \end{bmatrix}}_{2 \times 1} \quad Z^{[3]} = {\begin{bmatrix} {z_1}^{[3]^{(1)}} \\ {z_2}^{[3]^{(2)}} \end{bmatrix}}_{2 \times 1}$$

So, the Jacobian matrix of $A^{[3]}$ with respect to $Z^{[3]}$ is shape of $(2 \times 1) \times (1 \times 2) = (2 \times 2)$.

The Jacobian matrix is:

$$
\frac{\partial \, \text{vec}(A^{[3]})}{\partial \, \text{vec}(Z^{[3]})} = \text{Jacobian}_{A^{[3]}, Z^{[3]}} = \begin{bmatrix} \frac{\partial {a_1}^{[3]^{(1)}}}{\partial {z_1}^{[3]^{(1)}}} & \frac{\partial {a_1}^{[3]^{(1)}}}{\partial {z_2}^{[3]^{(2)}}} \\ \frac{\partial {a_2}^{[3]^{(2)}}}{\partial {z_1}^{[3]^{(1)}}} & \frac{\partial {a_2}^{[3]^{(2)}}}{\partial {z_2}^{[3]^{(2)}}} \end{bmatrix}$$

If we calculate each element of the Jacobian matrix we have:

$$\frac{\partial \, \text{vec}(A^{[3]})}{\partial \, \text{vec}(Z^{[3]})} = \begin{bmatrix} \sigma({z_1}^{[3]^{(1)}}) \cdot (1 - \sigma({z_1}^{[3]^{(1)}})) & 0 \\ 0 & \sigma({z_2}^{[3]^{(2)}}) \cdot (1 - \sigma({z_2}^{[3]^{(2)}})) \end{bmatrix}$$

Again here, for the elements which are not related we set the value to $0$. For example, the linear transformation of the first example ${z_1}^{[3]^{(1)}}$ does not affect the output of the second example ${a_2}^{[3]^{(2)}}$, and so on.

We know $\sigma(Z^{[3]}) = A^{[3]}$. So, we can rewrite the Jacobian matrix using the values of $A^{[3]}$:

$$
\frac{\partial \, \text{vec}(A^{[3]})}{\partial \, \text{vec}(Z^{[3]})} = \begin{bmatrix} {a_1}^{[3]^{(1)}} \cdot (1 - {a_1}^{[3]^{(1)}}) & 0 \\ 0 & {a_2}^{[3]^{(2)}} \cdot (1 - {a_2}^{[3]^{(2)}}) \end{bmatrix}$$
$$

Now if we write the chain rule in the vectorized form we have:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(Z^{[3]})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(A^{[3]})} \cdot \frac{\partial \,\text{vec}(A^{[3]})}{\partial \, \text{vec}(Z^{[3]})}
$$

And if we plug in the values we have:

$$\begin{bmatrix} 0.5501 & -0.6715 \end{bmatrix} \cdot \begin{bmatrix} 0.0828 & 0 \\ 0 & 0.1901 \end{bmatrix} = \begin{bmatrix} 0.0456 & -0.1277 \end{bmatrix}


Deflattening (de-vectorize) the result to the shape of the matrix which we were calculating the gradient with respect to (i.e. $Z^{[3]}$):

$$\frac{\partial J(\theta)}{\partial {Z^{[3]}}} = \begin{bmatrix} 0.0456 \\ -0.1277 \end{bmatrix}$$

### 4. Gradient of $J$ with respect to $W^{[3]}$ and ${\vec{\mathbf{b}}}^{[3]}$

Now we again go one step back to in the computational graph to calculate the gradient of the cost with respect to the weights and biases of the output layer (layer 3).

**Gradient of $J$ with respect to $\vec{\mathbf{b}}^{[3]}$:**


The linear transformation of the output layer is:

$$Z^{[3]} = A^{[2]} \cdot {W^{[3]}}^\top + {\vec{\mathbf{b}}}^{[3]}$$

We can write the chain rule as:

$$\frac{\partial J(\theta)}{\partial {\vec{\mathbf{b}}}^{[3]}} = \frac{\partial J(\theta)}{\partial {Z^{[3]}}} \cdot \frac{\partial {Z^{[3]}}}{\partial {\vec{\mathbf{b}}}^{[3]}}$$

We already calculated $\frac{\partial J(\theta)}{\partial {Z^{[3]}}}$ in the previous step. So, we just need to calculate $\frac{\partial {Z^{[3]}}}{\partial {\vec{\mathbf{b}}}^{[3]}}$.


$Z^{[3]}$ and ${\vec{\mathbf{b}}}^{[3]}$ are both matrices. So, again here we need to use derivative for Vector-valued Function of a Matrix again.

$$Z^{[3]} = {\begin{bmatrix} {z_1}^{[3]^{(1)}} \\ {z_2}^{[3]^{(2)}} \end{bmatrix}}_{2 \times 1} \quad {\vec{\mathbf{b}}}^{[3]} = {\begin{bmatrix} {b_{1}}^{[3]}  \end{bmatrix}}_{1 \times 1}$$

$\vec{\mathbf{b}}^{[3]}$ is a vector of biases in the output layer. We have only one neuron in the output layer. So, the bias vector is a vector with one element.
> Note: Weights and biases are in dependent of the samples. In other words, the number of rows in $Z$ and $A$ is equal to the number of samples in the batch and we indicate the $i$-th sample with the superscript $(i)$. But the weights and biases are independent of the samples. So, we don't use the superscript $(i)$ for the weights and biases.

The Jacobian matrix of $Z^{[3]}$ with respect to ${\vec{\mathbf{b}}}^{[3]}$ is shape of $(2 \times 1) \times (1 \times 1) = (2 \times 1)$.

The Jacobian matrix is:

$$
\frac{\partial \, \text{vec}(Z^{[3]})}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[3]})} = \text{Jacobian}_{Z^{[3]}, {\vec{\mathbf{b}}}^{[3]}} = \begin{bmatrix} \frac{\partial {z_1}^{[3]^{(1)}}}{\partial {b_{1}}^{[3]}} \\ \frac{\partial {z_2}^{[3]^{(2)}}}{\partial {b_{1}}^{[3]}} \end{bmatrix}
$$


Now if we write the chain rule in the vectorized form we have:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[3]})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(Z^{[3]})} \cdot \frac{\partial \,\text{vec}(Z^{[3]})}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[3]})}
$$

So, if we plug in the values we have:

$$
\begin{bmatrix} 0.0456 & -0.1277 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} -0.0821 \end{bmatrix}
$$

De-vectorizing (deflattening) the result to the shape of ${\vec{\mathbf{b}}}^{[3]}$, is the same as the result:

$$\frac{\partial J(\theta)}{\partial {\vec{\mathbf{b}}}^{[3]}} = \begin{bmatrix} -0.0821 \end{bmatrix}$$


Let's verify our calculation with PyTorch.


```python
print(f"Gradient for b3:\n{model.linear3.bias.grad}")
```

    Gradient for b3:
    tensor([-0.0821])


**Gradient of $J$ with respect to $W^{[3]}$:**

The linear transformation of the output layer is:

$$Z^{[3]} = A^{[2]} \cdot {W^{[3]}}^\top + {\vec{\mathbf{b}}}^{[3]}$$


Using the chain rule we can write:

$$\frac{\partial J(\theta)}{\partial {W^{[3]}}} = \frac{\partial J(\theta)}{\partial {Z^{[3]}}} \cdot \frac{\partial {Z^{[3]}}}{\partial {W^{[3]}}}$$

Again here, we already calculated $\frac{\partial J(\theta)}{\partial {Z^{[3]}}}$ in the previous step. So, we just need to calculate $\frac{\partial {Z^{[3]}}}{\partial {W^{[3]}}}$.

$Z^{[3]}$ and ${W^{[3]}}$ are both matrices:


$$Z^{[3]} = {\begin{bmatrix} {z_1}^{[3]^{(1)}} \\ {z_2}^{[3]^{(2)}} \end{bmatrix}}_{2 \times 1} \quad W^{[3]} = {\begin{bmatrix} w_{11}^{[3]} & w_{12}^{[3]} \end{bmatrix}}_{1 \times 2}$$


$W^{[3]}$ is a matrix of weights in the output layer.
- We have one neuron in layer 3 (output layer). So, $W$ has only one row.
- $w_{11}^{[3]}$ is the weight of the first input of the neuron
- $w_{12}^{[3]}$ is the weight of the second input of the neuron


The Jacobian matrix of $Z^{[3]}$ with respect to ${W^{[3]}}$ is shape of $(2 \times 1) \times (2 \times 1) = (2 \times 2)$.

The Jacobian matrix is:

$$
\frac{\partial \, \text{vec}(Z^{[3]})}{\partial \, \text{vec}({W^{[3]}})} = \text{Jacobian}_{Z^{[3]}, {W^{[3]}}} = \begin{bmatrix} \frac{\partial {z_1}^{[3]^{(1)}}}{\partial w_{11}^{[3]}} & \frac{\partial {z_1}^{[3]^{(1)}}}{\partial w_{12}^{[3]}} \\ \frac{\partial {z_2}^{[3]^{(2)}}}{\partial w_{11}^{[3]}} & \frac{\partial {z_2}^{[3]^{(2)}}}{\partial w_{12}^{[3]}} \end{bmatrix}
$$


If we calculate the partial derivatives of each element of the Jacobian matrix we have:

$$
\frac{\partial \, \text{vec}(Z^{[3]})}{\partial \, \text{vec}({W^{[3]}})} = \begin{bmatrix} {a_{11}}^{[2]^{(1)}} & {a_{12}}^{[2]^{(1)}} \\ {a_{11}}^{[2]^{(2)}} & {a_{12}}^{[2]^{(2)}} \end{bmatrix}$$


Where ${{a_{11}}^{[2]}}^{(i)}$ is the first input of the first row of $A^{[2]}$ for the $i$-th sample, and ${{a_{12}}^{[2]}}^{(i)}$ is the second output of the first row of $A^{[2]}$ for the $i$-th sample.


```python
print(f"A2:\n{model_results["A2"]}")
```

    A2:
    tensor([[ 0.0000,  8.0000],
            [ 8.0000, 10.1000]], grad_fn=<ReluBackward0>)


Putting the values of $A^{[2]}$ we have:

$$
\frac{\partial \, \text{vec}(Z^{[3]})}{\partial \, \text{vec}({W^{[3]}})} = \begin{bmatrix} 0 & 8 \\ 8 & 10.1 \end{bmatrix}
$$


Now if we write the chain rule in the vectorized form we have:
$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}({W^{[3]}})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(Z^{[3]})} \cdot \frac{\partial \,\text{vec}(Z^{[3]})}{\partial \, \text{vec}({W^{[3]}})}
$$

If we plug in the values we have:

$$
\begin{bmatrix} 0.0456 & -0.1277 \end{bmatrix} \cdot \begin{bmatrix} 0 & 8 \\ 8 & 10.1 \end{bmatrix} = \begin{bmatrix} -1.0216 & -0.9250 \end{bmatrix}
$$

The shape of the result is $(1 \times 2)$ which is the same as the shape of $W^{[3]}$. So, deflattening the result would result in the same matrix.

$$
\frac{\partial J(\theta)}{\partial {W^{[3]}}} = \begin{bmatrix} -1.0216 & -0.9250 \end{bmatrix}
$$

Let's verify our calculation with PyTorch.


```python
print(f"Gradient for W3:\n{model.linear3.weight.grad}")
```

    Gradient for W3:
    tensor([[-1.0216, -0.9253]])


> Since in our calculation we round number to 4 decimal places, we may have some small difference when comparing with the PyTorch result. We use the PyTorch result as the ground truth when calculating the next steps to avoid too much rounding error.

### 5. Gradient of $J$ with respect to $A^{[2]}$
Now, we go one step back in the computational graph to calculate the gradient of the cost with respect to the output of the layer 2. The steps are similar to the previous steps.

The linear transformation of the layer 3 we had:

$$Z^{[3]} = A^{[2]} \cdot {W^{[3]}}^\top + {\vec{\mathbf{b}}}^{[3]}$$

So, using the chain rule we can write:

$$\frac{\partial J(\theta)}{\partial {A^{[2]}}} = \frac{\partial J(\theta)}{\partial {Z^{[3]}}} \cdot \frac{\partial {Z^{[3]}}}{\partial {A^{[2]}}}$$

We already calculated $\frac{\partial J(\theta)}{\partial {Z^{[3]}}}$ in the previous step. So, we just need to calculate $\frac{\partial {Z^{[3]}}}{\partial {A^{[2]}}}$.

$Z^{[3]}$ and $A^{[2]}$ are both matrices. So, similar to the previous steps we use Jacobian matrix.

$$Z^{[3]} = {\begin{bmatrix} {z_1}^{[3]^{(1)}} \\ {z_2}^{[3]^{(2)}} \end{bmatrix}}_{2 \times 1} \quad A^{[2]} = {\begin{bmatrix} {a_{11}^{[2]}}^{(1)} & {a_{12}^{[2]}}^{(1)} \\ {a_{21}^{[2]}}^{(2)} & {a_{22}^{[2]}}^{(2)} \end{bmatrix}}_{2 \times 2}$$

So, the Jacobian matrix of $Z^{[3]}$ with respect to $A^{[2]}$ is shape of $(2 \times 1) \times (2 \times 2) = (2 \times 4)$.

The Jacobian matrix is:

$$
\frac{\partial \, \text{vec}(Z^{[3]})}{\partial \, \text{vec}(A^{[2]})} = \begin{bmatrix} \frac{\partial {z_1}^{[3]^{(1)}}}{\partial {a_{11}^{[2]}}^{(1)}} & \frac{\partial {z_1}^{[3]^{(1)}}}{\partial {a_{12}^{[2]}}^{(1)}} & \frac{\partial {z_1}^{[3]^{(1)}}}{\partial {a_{21}^{[2]}}^{(2)}} & \frac{\partial {z_1}^{[3]^{(1)}}}{\partial {a_{22}^{[2]}}^{(2)}} \\ \frac{\partial {z_2}^{[3]^{(2)}}}{\partial {a_{11}^{[2]}}^{(1)}} & \frac{\partial {z_2}^{[3]^{(2)}}}{\partial {a_{12}^{[2]}}^{(1)}} & \frac{\partial {z_2}^{[3]^{(2)}}}{\partial {a_{21}^{[2]}}^{(2)}} & \frac{\partial {z_2}^{[3]^{(2)}}}{\partial {a_{22}^{[2]}}^{(2)}} \end{bmatrix}
$$

Let's write the $z$ function and calculate each element of the Jacobian matrix.

$${z_1}^{[3]^{(1)}} = {a_{11}^{[2]}}^{(1)} \cdot w_{11}^{[3]} + {a_{12}^{[2]}}^{(1)} \cdot w_{12}^{[3]} + {b_{1}}^{[3]}$$
$${z_2}^{[3]^{(2)}} = {a_{21}^{[2]}}^{(2)} \cdot w_{11}^{[3]} + {a_{22}^{[2]}}^{(2)} \cdot w_{12}^{[3]} + {b_{1}}^{[3]}$$

Now let's calculate the partial derivatives of each element of the Jacobian matrix. For the first row:

$$\frac{\partial {z_1}^{[3]^{(1)}}}{\partial {a_{11}^{[2]}}^{(1)}} = w_{11}^{[3]} \quad \frac{\partial {z_1}^{[3]^{(1)}}}{\partial {a_{12}^{[2]}}^{(1)}} = w_{12}^{[3]} \quad \frac{\partial {z_1}^{[3]^{(1)}}}{\partial {a_{21}^{[2]}}^{(2)}} = 0 \quad \frac{\partial {z_1}^{[3]^{(1)}}}{\partial {a_{22}^{[2]}}^{(2)}} = 0$$

Partial derivative of ${z_1}^{[3]^{(1)}}$ with respect to $a_{21}^{[2]}$ is $0$ because the linear transformation of the first example $z_1^{[3]^{(1)}}$ does not depend on (effected by) the output of the second example $a_{21}^{[2]}$. This is also true for partial derivative of ${z_1}^{[3]^{(1)}}$ with respect to $a_{22}^{[2]}$.


And for the second row:

$$\frac{\partial {z_2}^{[3]^{(2)}}}{\partial {a_{11}^{[2]}}^{(1)}} = 0 \quad \frac{\partial {z_2}^{[3]^{(2)}}}{\partial {a_{12}^{[2]}}^{(1)}} = 0 \quad \frac{\partial {z_2}^{[3]^{(2)}}}{\partial {a_{21}^{[2]}}^{(2)}} = w_{11}^{[3]} \quad \frac{\partial {z_2}^{[3]^{(2)}}}{\partial {a_{22}^{[2]}}^{(2)}} = w_{12}^{[3]}$$

So, the Jacobian matrix is:
$$
\frac{\partial \, \text{vec}(Z^{[3]})}{\partial \, \text{vec}(A^{[2]})} = \begin{bmatrix} w_{11}^{[3]} & w_{12}^{[3]} & 0 & 0 \\ 0 & 0 & w_{11}^{[3]} & w_{12}^{[3]} \end{bmatrix}
$$


If we write the chain rule in the vectorized form we have:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(A^{[2]})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(Z^{[3]})} \cdot \frac{\partial \,\text{vec}(Z^{[3]})}{\partial \, \text{vec}(A^{[2]})}
$$

And if we plug in the values we have:

$$\begin{bmatrix} 0.0456 & -0.1277 \end{bmatrix} \cdot \begin{bmatrix} 0.5 & -0.3 & 0 & 0 \\ 0 & 0 & 0.5 & -0.3 \end{bmatrix}$$

Which is equal to:

$$\begin{bmatrix} 0.0228 & -0.0137 & -0.0639 & 0.0383 \end{bmatrix}$$

And when we de-vectorize the result to the shape of $A^{[2]}$ we have:

$$\frac{\partial J(\theta)}{\partial {A^{[2]}}} = \begin{bmatrix} 0.0228 & -0.0137 \\ -0.0639 & 0.0383 \end{bmatrix}$$

### 6. Gradient of $J$ with respect to $Z^{[2]}$
Now we go one step back in the computational graph to calculate the gradient of the cost with respect to the linear transformation of the layer 2.



Using the chain rule we can write:

$$\frac{\partial J(\theta)}{\partial {Z^{[2]}}} = \frac{\partial J(\theta)}{\partial {A^{[2]}}} \cdot \frac{\partial {A^{[2]}}}{\partial {Z^{[2]}}}$$

We already calculated $\frac{\partial J(\theta)}{\partial {A^{[2]}}}$ in the previous step. So, we just need to calculate $\frac{\partial {A^{[2]}}}{\partial {Z^{[2]}}}$.


$A^{[2]}$ is a function of $Z^{[2]}$ through the ReLU activation function.

$$A^{[2]} = \text{ReLU}(Z^{[2]})$$


ReLU activation function is defined as:

$$\text{ReLU}(x) = \begin{cases}
    x & \text{if $x>0$}\\
    0 & \text{if $x\leq0$}
  \end{cases}$$

Derivative of ReLU function is as follows:
$$\frac{d}{dx}ReLU(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

>Note: Derivative of ReLU function is not defined at $x=0$. But in practive, we can set it to $0$ or $1$. In this example we set it to $0$.

In calculation of the partial derivative of $A^{[2]}$ with respect to $Z^{[2]}$, we need to use the Jacobian matrix again as both $A^{[2]}$ and $Z^{[2]}$ are matrices.

$$A^{[2]} = {\begin{bmatrix} {a_{11}^{[2]}}^{(1)} & {a_{12}^{[2]}}^{(1)} \\ {a_{21}^{[2]}}^{(2)} & {a_{22}^{[2]}}^{(2)} \end{bmatrix}}_{2 \times 2} \quad Z^{[2]} = {\begin{bmatrix} {z_{11}^{[2]}}^{(1)} & {z_{12}^{[2]}}^{(1)} \\ {z_{21}^{[2]}}^{(2)} & {z_{22}^{[2]}}^{(2)} \end{bmatrix}}_{2 \times 2}$$

So, the Jacobian matrix of $A^{[2]}$ with respect to $Z^{[2]}$ is shape of $(2 \times 2) \times (2 \times 2) = (4 \times 4)$.

The Jacobian matrix is:

$$
\frac{\partial \, \text{vec}(A^{[2]})}{\partial \, \text{vec}(Z^{[2]})} = \begin{bmatrix} \frac{\partial {a_{11}^{[2]}}^{(1)}}{\partial {z_{11}^{[2]}}^{(1)}} & \frac{\partial {a_{11}^{[2]}}^{(1)}}{\partial {z_{12}^{[2]}}^{(1)}} & \frac{\partial {a_{11}^{[2]}}^{(1)}}{\partial {z_{21}^{[2]}}^{(2)}} & \frac{\partial {a_{11}^{[2]}}^{(1)}}{\partial {z_{22}^{[2]}}^{(2)}} \\ \frac{\partial {a_{12}^{[2]}}^{(1)}}{\partial {z_{11}^{[2]}}^{(1)}} & \frac{\partial {a_{12}^{[2]}}^{(1)}}{\partial {z_{12}^{[2]}}^{(1)}} & \frac{\partial {a_{12}^{[2]}}^{(1)}}{\partial {z_{21}^{[2]}}^{(2)}} & \frac{\partial {a_{12}^{[2]}}^{(1)}}{\partial {z_{22}^{[2]}}^{(2)}} \\ \frac{\partial {a_{21}^{[2]}}^{(2)}}{\partial {z_{11}^{[2]}}^{(1)}} & \frac{\partial {a_{21}^{[2]}}^{(2)}}{\partial {z_{12}^{[2]}}^{(1)}} & \frac{\partial {a_{21}^{[2]}}^{(2)}}{\partial {z_{21}^{[2]}}^{(2)}} & \frac{\partial {a_{21}^{[2]}}^{(2)}}{\partial {z_{22}^{[2]}}^{(2)}} \\ \frac{\partial {a_{22}^{[2]}}^{(2)}}{\partial {z_{11}^{[2]}}^{(1)}} & \frac{\partial {a_{22}^{[2]}}^{(2)}}{\partial {z_{12}^{[2]}}^{(1)}} & \frac{\partial {a_{22}^{[2]}}^{(2)}}{\partial {z_{21}^{[2]}}^{(2)}} & \frac{\partial {a_{22}^{[2]}}^{(2)}}{\partial {z_{22}^{[2]}}^{(2)}}
\end{bmatrix}$$

We have already calculated $Z^{[2]}$ in the forward propagation steps.

$$Z^{[2]} = \begin{bmatrix} 0 & 8 \\ 8 & 10.1 \end{bmatrix}$$


Now, we set the elements which are not related to $0$. Obviously, the linear transformation of the example $1$ has no effect on the output of the example $2$ and vice versa. So, we have:

$$
\frac{\partial \, \text{vec}(A^{[2]})}{\partial \, \text{vec}(Z^{[2]})} = \begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

For the elements which are related, we set the value to $1$ or $0$ based on the derivative of the ReLU function. For example, the first element of the Jacobian matrix is $0$ because ${z_{11}^{[2]}}^{(1)}=0$. So, the derivative of ReLU function is $0$.


The vectorized form of the chain rule is:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(Z^{[2]})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(A^{[2]})} \cdot \frac{\partial \,\text{vec}(A^{[2]})}{\partial \, \text{vec}(Z^{[2]})}
$$

If we plug in the values we have:

$$\begin{bmatrix} 0.0228 & -0.0137 & -0.0639 & 0.0383 \end{bmatrix} \cdot \begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

Which is equal to:

$$\begin{bmatrix} 0 & -0.0137 & -0.0639 & 0.0383 \end{bmatrix}$$

De-vectorizing (deflattening) the result to the shape of $Z^{[2]}$:

$$\frac{\partial J(\theta)}{\partial {Z^{[2]}}} = \begin{bmatrix} 0 & -0.0137 \\ -0.0639 & 0.0383 \end{bmatrix}$$

### 7. Gradient of $J$ with respect to $W^{[2]}$ and ${\vec{\mathbf{b}}}^{[2]}$

Now we reached the parameters of the layer 2. Similar to what we did for the layer 3, we calculate the gradient (partial derivative) of the cost with respect to the weights and biases of the layer 2.

**Gradient of $J$ with respect to $\vec{\mathbf{b}}^{[2]}$:**


Using the chain rule and the linear transformation of the layer 2 we can write:

$$\frac{\partial J(\theta)}{\partial {\vec{\mathbf{b}}}^{[2]}} = \frac{\partial J(\theta)}{\partial {Z^{[2]}}} \cdot \frac{\partial {Z^{[2]}}}{\partial {\vec{\mathbf{b}}}^{[2]}}$$

We already calculated $\frac{\partial J(\theta)}{\partial {Z^{[2]}}}$ in the previous step. So, we just need to calculate $\frac{\partial {Z^{[2]}}}{\partial {\vec{\mathbf{b}}}^{[2]}}$.


$Z^{[2]}$ and ${\vec{\mathbf{b}}}^{[2]}$ are both matrices. So, similar to the previous steps we use Jacobian matrix.

$$Z^{[2]} = {\begin{bmatrix} {z_{11}^{[2]}}^{(1)} & {z_{12}^{[2]}}^{(1)} \\ {z_{21}^{[2]}}^{(2)} & {z_{22}^{[2]}}^{(2)} \end{bmatrix}}_{2 \times 2} \quad {\vec{\mathbf{b}}}^{[2]} = {\begin{bmatrix} {b_{11}}^{[2]} & {b_{12}}^{[2]} \end{bmatrix}}_{1 \times 2}$$

The Jacobian matrix of $Z^{[2]}$ with respect to ${\vec{\mathbf{b}}}^{[2]}$ is shape of $(2 \times 2) \times (1 \times 2) = (4 \times 2)$.

The Jacobian matrix is:
$$
\frac{\partial \, \text{vec}(Z^{[2]})}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[2]})} = \begin{bmatrix} \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial {b_{11}}^{[2]}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial {b_{12}}^{[2]}} \\ \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial {b_{11}}^{[2]}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial {b_{12}}^{[2]}} \\ \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial {b_{11}}^{[2]}} & \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial {b_{12}}^{[2]}} \\ \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial {b_{11}}^{[2]}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial {b_{12}}^{[2]}}
\end{bmatrix}$$

If we calculate the partial derivatives of each element of the Jacobian matrix we have:

$$\frac{\partial \, \text{vec}(Z^{[2]})}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[2]})} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$$

If we write the chain rule in the vectorized form we have:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[2]})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(Z^{[2]})} \cdot \frac{\partial \,\text{vec}(Z^{[2]})}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[2]})}
$$

If we plug in the values we have:

$$\begin{bmatrix} 0 & -0.0137 & -0.0639 & 0.0383 \end{bmatrix} \cdot \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$$

Which is equal to:

$$\begin{bmatrix} -0.0639 & 0.0246 \end{bmatrix}$$

De-vectorizing (deflattening) the result to the shape of ${\vec{\mathbf{b}}}^{[2]}$ is the same as the result itself as the shape of ${\vec{\mathbf{b}}}^{[2]}$ is $(1 \times 2)$.

$$\frac{\partial J(\theta)}{\partial {\vec{\mathbf{b}}}^{[2]}} = \begin{bmatrix} -0.0639 & 0.0246 \end{bmatrix}$$

Let's verify our calculation with PyTorch.


```python
print(f"Gradient for b2:\n{model.linear2.bias.grad}")
```

    Gradient for b2:
    tensor([-0.0639,  0.0246])


**Gradient of $J$ with respect to $W^{[2]}$:**

The linear transformation of the layer 2 is:
$$Z^{[2]} = A^{[1]} \cdot {W^{[2]}}^\top + {\vec{\mathbf{b}}}^{[2]}$$

Using the chain rule we can write:

$$\frac{\partial J(\theta)}{\partial {W^{[2]}}} = \frac{\partial J(\theta)}{\partial {Z^{[2]}}} \cdot \frac{\partial {Z^{[2]}}}{\partial {W^{[2]}}}$$

$Z^{[2]}$ and ${W^{[2]}}$ are both matrices:

$$Z^{[2]} = {\begin{bmatrix} {z_{11}^{[2]}}^{(1)} & {z_{12}^{[2]}}^{(1)} \\ {z_{21}^{[2]}}^{(2)} & {z_{22}^{[2]}}^{(2)} \end{bmatrix}}_{2 \times 2} \quad W^{[2]} = {\begin{bmatrix} w_{11}^{[2]} & w_{12}^{[2]} & w_{13}^{[2]} \\ w_{21}^{[2]} & w_{22}^{[2]} & w_{23}^{[2]} \end{bmatrix}}_{2 \times 3}$$

$W^{[2]}$ is a matrix of weights in the layer 2.
- Each row of $W^{[2]}$ is the weights of a neuron in the layer 2.
- Layer 2 has 2 neurons and 3 input features in every example. So, each neuron has 3 weights (one for each input feature).
- For example, $w_{11}^{[2]}, w_{12}^{[2]}, w_{13}^{[2]}$ are the weights of the first neuron in the layer 2.

The Jacobian matrix of $Z^{[2]}$ with respect to ${W^{[2]}}$ is shape of $(2 \times 2) \times (2 \times 3) = (4 \times 6)$.

The Jacobian matrix is:

$$
\frac{\partial \, \text{vec}(Z^{[2]})}{\partial \, \text{vec}({W^{[2]}})} = \begin{bmatrix} \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial w_{11}^{[2]}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial w_{12}^{[2]}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial w_{13}^{[2]}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial w_{21}^{[2]}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial w_{22}^{[2]}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial w_{23}^{[2]}} \\ \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial w_{11}^{[2]}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial w_{12}^{[2]}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial w_{13}^{[2]}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial w_{21}^{[2]}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial w_{22}^{[2]}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial w_{23}^{[2]}} \\ \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial w_{11}^{[2]}} & \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial w_{12}^{[2]}} & \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial w_{13}^{[2]}} & \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial w_{21}^{[2]}} & \frac{\partial {z_{21}^{[
2]}}^{(2)}}{\partial w_{22}^{[2]}} & \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial w_{23}^{[2]}} \\ \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial w_{11}^{[2]}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial w_{12}^{[2]}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial w_{13}^{[2]}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial w_{21}^{[2]}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial w_{22}^{[2]}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial w_{23}^{[2]}}
\end{bmatrix}
$$

Let's write the $z$ function and calculate each element of the Jacobian matrix.

$${z_{11}^{[2]}}^{(1)} = {a_{11}^{[1]}}^{(1)} \cdot w_{11}^{[2]} + {a_{12}^{[1]}}^{(1)} \cdot w_{12}^{[2]} + {a_{13}^{[1]}}^{(1)} \cdot w_{13}^{[2]} + {b_{11}}^{[2]}$$

$${z_{12}^{[2]}}^{(1)} = {a_{11}^{[1]}}^{(1)} \cdot w_{21}^{[2]} + {a_{12}^{[1]}}^{(1)} \cdot w_{22}^{[2]} + {a_{13}^{[1]}}^{(1)} \cdot w_{23}^{[2]} + {b_{12}}^{[2]}$$

$${z_{21}^{[2]}}^{(2)} = {a_{21}^{[1]}}^{(2)} \cdot w_{11}^{[2]} + {a_{22}^{[1]}}^{(2)} \cdot w_{12}^{[2]} + {a_{23}^{[1]}}^{(2)} \cdot w_{13}^{[2]} + {b_{11}}^{[2]}$$

$${z_{22}^{[2]}}^{(2)} = {a_{21}^{[1]}}^{(2)} \cdot w_{21}^{[2]} + {a_{22}^{[1]}}^{(2)} \cdot w_{22}^{[2]} + {a_{23}^{[1]}}^{(2)} \cdot w_{23}^{[2]} + {b_{12}}^{[2]}$$


If we calculate the partial derivatives of each element of the Jacobian matrix we have:

$$
\frac{\partial \, \text{vec}(Z^{[2]})}{\partial \, \text{vec}({W^{[2]}})} = \begin{bmatrix} {a_{11}^{[1]}}^{(1)} & {a_{12}^{[1]}}^{(1)} & {a_{13}^{[1]}}^{(1)} & 0 & 0 & 0 \\ 0 & 0 & 0 & {a_{11}^{[1]}}^{(1)} & {a_{12}^{[1]}}^{(1)} & {a_{13}^{[1]}}^{(1)} \\ {a_{21}^{[1]}}^{(2)} & {a_{22}^{[1]}}^{(2)} & {a_{23}^{[1]}}^{(2)} & 0 & 0 & 0 \\ 0 & 0 & 0 & {a_{21}^{[1]}}^{(2)} & {a_{22}^{[1]}}^{(2)} & {a_{23}^{[1]}}^{(2)} \end{bmatrix}
$$

Again here, some of the elements are $0$ because the variable which we are calculating the partial derivative with respect to, does not affect the output of the function. For example, output of first neuron $z_{11}^{[2]}$ does not depend on the parameters of the second neuron. So, the partial derivative of $z_{11}^{[2]}$ with respect to $w_{21}^{[2]}, w_{22}^{[2]}, w_{23}^{[2]}$ is $0$.

We have calculated the values of $A^{[1]}$ in the forward propagation steps.


```python
print(f"A1:\n{model_results["A1"]}")
```

    A1:
    tensor([[4., 2., 0.],
            [6., 9., 0.]], grad_fn=<ReluBackward0>)


If we write the chain rule in the vectorized form we have:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}({W^{[2]}})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(Z^{[2]})} \cdot \frac{\partial \,\text{vec}(Z^{[2]})}{\partial \, \text{vec}({W^{[2]}})}
$$

If we plug in the values we have:

$$\begin{bmatrix} 0 & -0.0137 & -0.0639 & 0.0383 \end{bmatrix} \cdot \begin{bmatrix} 4 & 2 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 4 & 2 & 0 \\ 6 & 9 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 6 & 9 & 0 \end{bmatrix}$$



Which is equal to:

$$\begin{bmatrix} -0.3831 & -0.5747 & 0 & 0.1752 & 0.3173 & 0 \end{bmatrix}$$

De-vectorizing (deflattening) the result to the shape of $W^{[2]}$:

$$\frac{\partial J(\theta)}{\partial {W^{[2]}}} = \begin{bmatrix} -0.3831 & -0.5747 & 0 \\ 0.1752 & 0.3173 & 0 \end{bmatrix}$$

Let's verify our calculation with PyTorch.


```python
print(f"Gradient for W2:\n{model.linear2.weight.grad}")
```

    Gradient for W2:
    tensor([[-0.3831, -0.5747,  0.0000],
            [ 0.1752,  0.3175,  0.0000]])


### 8. Gradient of $J$ with respect to $A^{[1]}$

We go one step back in the computational graph to calculate the gradient of the cost with respect to the output of the layer 1.

The linear transformation of the layer 2 we had:

$$Z^{[2]} = A^{[1]} \cdot {W^{[2]}}^\top + {\vec{\mathbf{b}}}^{[2]}$$

Using the chain rule we can write:

$$\frac{\partial J(\theta)}{\partial {A^{[1]}}} = \frac{\partial J(\theta)}{\partial {Z^{[2]}}} \cdot \frac{\partial {Z^{[2]}}}{\partial {A^{[1]}}}$$

We already calculated $\frac{\partial J(\theta)}{\partial {Z^{[2]}}}$ in the previous step. So, we just need to calculate $\frac{\partial {Z^{[2]}}}{\partial {A^{[1]}}}$.

$Z^{[2]}$ and $A^{[1]}$ are both matrices. So, similar to the previous steps we use Jacobian matrix.

$$Z^{[2]} = {\begin{bmatrix} {z_{11}^{[2]}}^{(1)} & {z_{12}^{[2]}}^{(1)} \\ {z_{21}^{[2]}}^{(2)} & {z_{22}^{[2]}}^{(2)} \end{bmatrix}}_{2 \times 2} \quad A^{[1]} = {\begin{bmatrix} {a_{11}^{[1]}}^{(1)} & {a_{12}^{[1]}}^{(1)} & {a_{13}^{[1]}}^{(1)} \\ {a_{21}^{[1]}}^{(2)} & {a_{22}^{[1]}}^{(2)} & {a_{23}^{[1]}}^{(2)} \end{bmatrix}}_{2 \times 3}$$

The Jacobian matrix of $Z^{[2]}$ with respect to $A^{[1]}$ is shape of $(2 \times 2) \times (2 \times 3) = (4 \times 6)$.

The Jacobian matrix is:

$$
\frac{\partial \, \text{vec}(Z^{[2]})}{\partial \, \text{vec}(A^{[1]})} = \begin{bmatrix} \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial {a_{11}^{[1]}}^{(1)}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial {a_{12}^{[1]}}^{(1)}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial {a_{13}^{[1]}}^{(1)}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial {a_{21}^{[1]}}^{(2)}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial {a_{22}^{[1]}}^{(2)}} & \frac{\partial {z_{11}^{[2]}}^{(1)}}{\partial {a_{23}^{[1]}}^{(2)}} \\ \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial {a_{11}^{[1]}}^{(1)}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial {a_{12}^{[1]}}^{(1)}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial {a_{13}^{[1]}}^{(1)}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial {a_{21}^{[1]}}^{(2)}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial {a_{22}^{[1]}}^{(2)}} & \frac{\partial {z_{12}^{[2]}}^{(1)}}{\partial {a_{23}^{[1]}}^{(2)}} \\ \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial {a_{11}^{[1]}}^{(1)}} & \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial {a_{12}^{[1]}}^{(1)}} & \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial {a_{13}^{[1]}}^{(1)}} & \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial {a_{21}^{[1]}}^{(2)}} & \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial {a_{22}^{[1]}}^{(2)}} & \frac{\partial {z_{21}^{[2]}}^{(2)}}{\partial {a_{23}^{[1]}}^{(2)}} \\ \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial {a_{11}^{[1]}}^{(1)}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial {a_{12}^{[1]}}^{(1)}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial {a_{13}^{[1]}}^{(1)}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial {a_{21}^{[1]}}^{(2)}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial {a_{22}^{[1]}}^{(2)}} & \frac{\partial {z_{22}^{[2]}}^{(2)}}{\partial {a_{23}^{[1]}}^{(2)}}
\end{bmatrix}
$$

Let's write the $z$ function and calculate each element of the Jacobian matrix.

$${z_{11}^{[2]}}^{(1)} = {a_{11}^{[1]}}^{(1)} \cdot w_{11}^{[2]} + {a_{12}^{[1]}}^{(1)} \cdot w_{12}^{[2]} + {a_{13}^{[1]}}^{(1)} \cdot w_{13}^{[2]} + {b_{11}}^{[2]}$$

$${z_{12}^{[2]}}^{(1)} = {a_{11}^{[1]}}^{(1)} \cdot w_{21}^{[2]} + {a_{12}^{[1]}}^{(1)} \cdot w_{22}^{[2]} + {a_{13}^{[1]}}^{(1)} \cdot w_{23}^{[2]} + {b_{12}}^{[2]}$$

$${z_{21}^{[2]}}^{(2)} = {a_{21}^{[1]}}^{(2)} \cdot w_{11}^{[2]} + {a_{22}^{[1]}}^{(2)} \cdot w_{12}^{[2]} + {a_{23}^{[1]}}^{(2)} \cdot w_{13}^{[2]} + {b_{11}}^{[2]}$$

$${z_{22}^{[2]}}^{(2)} = {a_{21}^{[1]}}^{(2)} \cdot w_{21}^{[2]} + {a_{22}^{[1]}}^{(2)} \cdot w_{22}^{[2]} + {a_{23}^{[1]}}^{(2)} \cdot w_{23}^{[2]} + {b_{12}}^{[2]}$$

If we calculate the partial derivatives of each element of the Jacobian matrix we have:

$$
\frac{\partial \, \text{vec}(Z^{[2]})}{\partial \, \text{vec}(A^{[1]})} = \begin{bmatrix} w_{11}^{[2]} & w_{12}^{[2]} & w_{13}^{[2]} & 0 & 0 & 0 \\ w_{21}^{[2]} & w_{22}^{[2]} & w_{23}^{[2]} & 0 & 0 & 0 \\ 0 & 0 & 0 & w_{11}^{[2]} & w_{12}^{[2]} & w_{13}^{[2]} \\ 0 & 0 & 0 & w_{21}^{[2]} & w_{22}^{[2]} & w_{23}^{[2]} \end{bmatrix}
$$


```python
print(f"W2:\n{model.linear2.weight}")
```

    W2:
    Parameter containing:
    tensor([[ 0.5000,  1.0000, -2.0000],
            [ 0.7000,  0.1000,  0.3000]], requires_grad=True)


Writing the chain rule in the vectorized form we have:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}({A^{[1]}})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(Z^{[2]})} \cdot \frac{\partial \,\text{vec}(Z^{[2]})}{\partial \, \text{vec}({A^{[1]}})}
$$

If we plug in the values we have:

$$\begin{bmatrix} 0 & -0.0137 & -0.0639 & 0.0383 \end{bmatrix} \cdot \begin{bmatrix} 0.5 & 1 & -2 & 0 & 0 & 0 \\ 0.7 & 0.1 & 0.3 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0.5 & 1 & -2 \\ 0 & 0 & 0 & 0.7 & 0.1 & 0.3 \end{bmatrix}$$

Which is equal to:

$$\begin{bmatrix} -0.0096 & -0.0014 & -0.0041 & -0.0051 & -0.0601 & 0.1393 \end{bmatrix}$$

De-vectorizing (deflattening) the result to the shape of $A^{[1]}$:

$$\frac{\partial J(\theta)}{\partial {A^{[1]}}} = \begin{bmatrix} -0.0096 & -0.0014 & -0.0041 \\ -0.0051 & -0.0601 & 0.1393 \end{bmatrix}$$

### 9. Gradient of $J$ with respect to $Z^{[1]}$

We go one step back in the computational graph to calculate the gradient of the cost with respect to the linear transformation of the layer 1.

Using the chain rule we can write:

$$\frac{\partial J(\theta)}{\partial {Z^{[1]}}} = \frac{\partial J(\theta)}{\partial {A^{[1]}}} \cdot \frac{\partial {A^{[1]}}}{\partial {Z^{[1]}}}$$

We already calculated $\frac{\partial J(\theta)}{\partial {A^{[1]}}}$ in the previous step. So, we just need to calculate $\frac{\partial {A^{[1]}}}{\partial {Z^{[1]}}}$.

$A^{[1]}$ is a function of $Z^{[1]}$ through the ReLU activation function:

$$A^{[1]} = \text{ReLU}(Z^{[1]})$$

As we saw in calculating the backpropagation of the ReLU activation function for the layer 2, the derivative of the ReLU function is:

$$\frac{d}{dx}ReLU(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

$A^{[1]}$ and $Z^{[1]}$ are both matrices. So, similar to the previous steps we use Jacobian matrix.

$$A^{[1]} = {\begin{bmatrix} {a_{11}^{[1]}}^{(1)} & {a_{12}^{[1]}}^{(1)} & {a_{13}^{[1]}}^{(1)} \\ {a_{21}^{[1]}}^{(2)} & {a_{22}^{[1]}}^{(2)} & {a_{23}^{[1]}}^{(2)} \end{bmatrix}}_{2 \times 3} \quad Z^{[1]} = {\begin{bmatrix} {z_{11}^{[1]}}^{(1)} & {z_{12}^{[1]}}^{(1)} & {z_{13}^{[1]}}^{(1)} \\ {z_{21}^{[1]}}^{(2)} & {z_{22}^{[1]}}^{(2)} & {z_{23}^{[1]}}^{(2)} \end{bmatrix}}_{2 \times 3}$$

The Jacobian matrix of $A^{[1]}$ with respect to $Z^{[1]}$ is shape of $(2 \times 3) \times (2 \times 3) = (6 \times 6)$.

The Jacobian matrix is:

$$
\frac{\partial \, \text{vec}(A^{[1]})}{\partial \, \text{vec}(Z^{[1]})} = \begin{bmatrix} \frac{\partial {a_{11}^{[1]}}^{(1)}}{\partial {z_{11}^{[1]}}^{(1)}} & \frac{\partial {a_{11}^{[1]}}^{(1)}}{\partial {z_{12}^{[1]}}^{(1)}} & \dots & \frac{\partial {a_{11}^{[1]}}^{(1)}}{\partial {z_{23}^{[1]}}^{(1)}} \\ \frac{\partial {a_{12}^{[1]}}^{(1)}}{\partial {z_{11}^{[1]}}^{(1)}} & \frac{\partial {a_{12}^{[1]}}^{(1)}}{\partial {z_{12}^{[1]}}^{(1)}} & \dots & \frac{\partial {a_{12}^{[1]}}^{(1)}}{\partial {z_{23}^{[1]}}^{(1)}} \\ \frac{\partial {a_{13}^{[1]}}^{(1)}}{\partial {z_{11}^{[1]}}^{(1)}} & \frac{\partial {a_{13}^{[1]}}^{(1)}}{\partial {z_{12}^{[1]}}^{(1)}} & \dots & \frac{\partial {a_{13}^{[1]}}^{(1)}}{\partial {z_{23}^{[1]}}^{(1)}} \\ \frac{\partial {a_{21}^{[1]}}^{(2)}}{\partial {z_{11}^{[1]}}^{(2)}} & \frac{\partial {a_{21}^{[1]}}^{(2)}}{\partial {z_{12}^{[1]}}^{(2)}} & \dots & \frac{\partial {a_{21}^{[1]}}^{(2)}}{\partial {z_{23}^{[1]}}^{(2)}} \\ \frac{\partial {a_{22}^{[1]}}^{(2)}}{\partial {z_{11}^{[1]}}^{(2)}} & \frac{\partial {a_{22}^{[1]}}^{(2)}}{\partial {z_{12}^{[1]}}^{(2)}} & \dots & \frac{\partial {a_{22}^{[1]}}^{(2)}}{\partial {z_{23}^{[1]}}^{(2)}} \\ \frac{\partial {a_{23}^{[1]}}^{(2)}}{\partial {z_{11}^{[1]}}^{(2)}} & \frac{\partial {a_{23}^{[1]}}^{(2)}}{\partial {z_{12}^{[1]}} ^{(2)}} & \dots & \frac{\partial {a_{23}^{[1]}}^{(2)}}{\partial {z_{23}^{[1]}}^{(2)}}
\end{bmatrix}
$$


Calculating the partial derivatives of each element of the Jacobian matrix we have:

$$
\frac{\partial \, \text{vec}(A^{[1]})}{\partial \, \text{vec}(Z^{[1]})} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}
$$

The vectorized form of the chain rule is:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}({Z^{[1]}})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(A^{[1]})} \cdot \frac{\partial \,\text{vec}(A^{[1]})}{\partial \, \text{vec}({Z^{[1]}})}
$$

If we plug in the values and calculate the matrix multiplication we have:

$$\begin{bmatrix} -0.0096 & -0.0014 & 0 & -0.0051 & -0.0600 & 0 \end{bmatrix}$$

De-vectorizing (deflattening) the result to the shape of $Z^{[1]}$:

$$\frac{\partial J(\theta)}{\partial {Z^{[1]}}} = \begin{bmatrix} -0.0096 & -0.0014 & 0 \\ -0.0051 & -0.0600 & 0 \end{bmatrix}$$

### 10. Gradient of $J$ with respect to $W^{[1]}$ and ${\vec{\mathbf{b}}}^{[1]}$

We go one step back in the computational graph to calculate the gradient of the cost with respect to parameters of the layer 1.

**Gradient of $J$ with respect to $\vec{\mathbf{b}}^{[1]}$:**

Using the chain rule we can write:

$$\frac{\partial J(\theta)}{\partial {\vec{\mathbf{b}}}^{[1]}} = \frac{\partial J(\theta)}{\partial {Z^{[1]}}} \cdot \frac{\partial {Z^{[1]}}}{\partial {\vec{\mathbf{b}}}^{[1]}}$$

We already calculated $\frac{\partial J(\theta)}{\partial {Z^{[1]}}}$ in the previous step. So, we just need to calculate $\frac{\partial {Z^{[1]}}}{\partial {\vec{\mathbf{b}}}^{[1]}}$.

$Z^{[1]}$ and ${\vec{\mathbf{b}}}^{[1]}$ are both matrices. So, similar to the previous steps we use Jacobian matrix.

$$Z^{[1]} = {\begin{bmatrix} {z_{11}^{[1]}}^{(1)} & {z_{12}^{[1]}}^{(1)} & {z_{13}^{[1]}}^{(1)} \\ {z_{21}^{[1]}}^{(2)} & {z_{22}^{[1]}}^{(2)} & {z_{23}^{[1]}}^{(2)} \end{bmatrix}}_{2 \times 3} \quad {\vec{\mathbf{b}}}^{[1]} = {\begin{bmatrix} b_{11}^{[1]} & b_{12}^{[1]} & b_{13}^{[1]} \end{bmatrix}}_{1 \times 3}$$

The Jacobian matrix of $Z^{[1]}$ with respect to ${\vec{\mathbf{b}}}^{[1]}$ is shape of $(2 \times 3) \times (1 \times 3) = (6 \times 3)$.

The Jacobian matrix is:

$$
\frac{\partial \, \text{vec}(Z^{[1]})}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[1]})} = \begin{bmatrix} \frac{\partial {z_{11}^{[1]}}^{(1)}}{\partial b_{11}^{[1]}} & \frac{\partial {z_{11}^{[1]}}^{(1)}}{\partial b_{12}^{[1]}} & \frac{\partial {z_{11}^{[1]}}^{(1)}}{\partial b_{13}^{[1]}} \\ \frac{\partial {z_{12}^{[1]}}^{(1)}}{\partial b_{11}^{[1]}} & \frac{\partial {z_{12}^{[1]}}^{(1)}}{\partial b_{12}^{[1]}} & \frac{\partial {z_{12}^{[1]}}^{(1)}}{\partial b_{13}^{[1]}} \\ \vdots \\
\frac{\partial {z_{23}^{[1]}}^{(2)}}{\partial b_{11}^{[1]}} & \frac{\partial {z_{23}^{[1]}}^{(2)}}{\partial b_{12}^{[1]}} & \frac{\partial {z_{23}^{[1]}}^{(2)}}{\partial b_{13}^{[1]}} \end{bmatrix}
$$

Calculating the partial derivatives of each element of the Jacobian matrix we have:

$$
\frac{\partial \, \text{vec}(Z^{[1]})}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[1]})} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

The vectorized form of the chain rule is:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[1]})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(Z^{[1]})} \cdot \frac{\partial \,\text{vec}(Z^{[1]})}{\partial \, \text{vec}({\vec{\mathbf{b}}}^{[1]})}
$$

If we plug in the values and calculate the matrix multiplication we have:

$$\begin{bmatrix} -0.0147 & -0.0614 & 0 \end{bmatrix}$$

De-vectorizing (deflattening) the result to the shape of ${\vec{\mathbf{b}}}^{[1]}$ which is $1 \times 3$:

$$\frac{\partial J(\theta)}{\partial {\vec{\mathbf{b}}}^{[1]}} = \begin{bmatrix} -0.0147 & -0.0614 & 0 \end{bmatrix}$$

Let's verify our calculation with PyTorch.


```python
print(f"Gradient for b1:\n{model.linear1.bias.grad}")
```

    Gradient for b1:
    tensor([-0.0147, -0.0614,  0.0000])


**Gradient of $J$ with respect to $W^{[1]}$:**

The linear transformation of the layer 1:

$$Z^{[1]} = A^{[0]} \cdot {W^{[1]}}^\top + {\vec{\mathbf{b}}}^{[1]}$$

We know the $A^{[0]} = X$ is the input of the neural network. So, we can write:
$$Z^{[1]} = X \cdot {W^{[1]}}^\top + {\vec{\mathbf{b}}}^{[1]}$$

Using the chain rule we can write:

$$\frac{\partial J(\theta)}{\partial {W^{[1]}}} = \frac{\partial J(\theta)}{\partial {Z^{[1]}}} \cdot \frac{\partial {Z^{[1]}}}{\partial {W^{[1]}}}$$

$Z^{[1]}$ and $W^{[1]}$ are both matrices. So, similar to the previous steps we use Jacobian matrix.

$$Z^{[1]} = {\begin{bmatrix} {z_{11}^{[1]}}^{(1)} & {z_{12}^{[1]}}^{(1)} & {z_{13}^{[1]}}^{(1)} \\ {z_{21}^{[1]}}^{(2)} & {z_{22}^{[1]}}^{(2)} & {z_{23}^{[1]}}^{(2)} \end{bmatrix}}_{2 \times 3} \quad W^{[1]} = {\begin{bmatrix} w_{11}^{[1]} & w_{12}^{[1]} \\ w_{21}^{[1]} & w_{22}^{[1]} \\ w_{31}^{[1]} & w_{32}^{[1]} \end{bmatrix}}_{3 \times 2}$$

The Jacobian matrix of $Z^{[1]}$ with respect to $W^{[1]}$ is shape of $(2 \times 3) \times (2 \times 3) = (6 \times 6)$.

The Jacobian matrix is:

$$
\frac{\partial \, \text{vec}(Z^{[1]})}{\partial \, \text{vec}({W^{[1]}})} = \begin{bmatrix} \frac{\partial {z_{11}^{[1]}}^{(1)}}{\partial w_{11}^{[1]}} & \frac{\partial {z_{11}^{[1]}}^{(1)}}{\partial w_{12}^{[1]}} & \dots & \frac{\partial {z_{11}^{[1]}}^{(1)}}{\partial w_{32}^{[1]}} \\ \frac{\partial {z_{12}^{[1]}}^{(1)}}{\partial w_{11}^{[1]}} & \frac{\partial {z_{12}^{[1]}}^{(1)}}{\partial w_{12}^{[1]}} & \dots & \frac{\partial {z_{12}^{[1]}}^{(1)}}{\partial w_{32}^{[1]}} \\ \vdots & \vdots & & \vdots \\ \frac{\partial {z_{23}^{[1]}}^{(2)}}{\partial w_{11}^{[1]}} & \frac{\partial {z_{23}^{[1]}}^{(2)}}{\partial w_{12}^{[1]}} & \dots & \frac{\partial {z_{23}^{[1]}}^{(2)}}{\partial w_{32}^{[1]}} \end{bmatrix}
$$

If we calculate the partial derivatives of each element of the Jacobian matrix we have:

$$
\frac{\partial \, \text{vec}(Z^{[1]})}{\partial \, \text{vec}({W^{[1]}})} = \begin{bmatrix} x_{11}^{(1)} & x_{12}^{(1)} & 0 & 0 & 0 & 0 \\ 0 & 0 & x_{11}^{(1)} & x_{12}^{(1)} & 0 & 0 \\ 0 & 0 & 0 & 0 & x_{11}^{(1)} & x_{12}^{(1)} \\ x_{21}^{(2)} & x_{22}^{(2)} & 0 & 0 & 0 & 0 \\ 0 & 0 & x_{21}^{(2)} & x_{22}^{(2)} & 0 & 0 \\ 0 & 0 & 0 & 0 & x_{21}^{(2)} & x_{22}^{(2)} \end{bmatrix}
$$


```python
print(f"A0=X:\n{X}")
```

    A0=X:
    tensor([[1., 2.],
            [3., 4.]])


If we plug in the values we have:

$$\begin{bmatrix} 1 & 2 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 2 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 2 \\ 3 & 4 & 0 & 0 & 0 & 0 \\ 0 & 0 & 3 & 4 & 0 & 0 \\ 0 & 0 & 0 & 0 & 3 & 4 \end{bmatrix}$$

If we write the chain rule in the vectorized form we have:

$$
\frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}({W^{[1]}})} = \frac{\partial \, \text{vec}(J(\theta))}{\partial \, \text{vec}(Z^{[1]})} \cdot \frac{\partial \,\text{vec}(Z^{[1]})}{\partial \, \text{vec}({W^{[1]}})}
$$

If we plug in the values we have:

$$\begin{bmatrix} -0.0096 & -0.0014 & 0 & -0.0051 & -0.0600 & 0 \end{bmatrix} \cdot \begin{bmatrix} 1 & 2 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 2 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 2 \\ 3 & 4 & 0 & 0 & 0 & 0 \\ 0 & 0 & 3 & 4 & 0 & 0 \\ 0 & 0 & 0 & 0 & 3 & 4 \end{bmatrix}$$

Which is equal to:

$$\begin{bmatrix} -0.0249 & -0.0396 & -0.1814 & -0.2428 & 0 & 0 \end{bmatrix}$$


De-vectorizing (deflattening) the result to the shape of $W^{[1]}$:
$$\frac{\partial J(\theta)}{\partial {W^{[1]}}} = \begin{bmatrix} -0.0249 & -0.0396 \\ -0.1814 & -0.2428 \\ 0 & 0 \end{bmatrix}$$

Let's verify our calculation with PyTorch.


```python
print(f"Gradient for W1:\n{model.linear1.weight.grad}")
```

    Gradient for W1:
    tensor([[-0.0249, -0.0396],
            [-0.1814, -0.2428],
            [ 0.0000,  0.0000]])


So, we have calculated the gradients of the cost function with respect to all the parameters of the neural network using backpropagation by computing the gradients of the cost function from the last node in the computational graph to the first node in the computational graph.


In the [Gradient Descent](https://pooya.io/ai/gradient-descent-machine-learning/) algorithm, after calculating the gradients of the cost function with respect to the parameters, we update the parameters.



## Update the Parameters

As we discussed in the [Gradient Descent](https://pooya.io/ai/gradient-descent-machine-learning/) algorithm, we update the parameters of the neural network using the gradients of the cost function with respect to the parameters.

The update rule is:

$$
\theta = \theta - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta}$$

Where $\alpha$ is the learning rate, and $\theta$ is the parameter of the neural network.

Let's calculate the the new values fo the parameters of layer 1, $W^{[1]}$ and ${\vec{\mathbf{b}}}^{[1]}$, using the gradients we calculated in the previous steps.

The parameters of the layer 1 are:

$$
W^{[1]} = \begin{bmatrix} -1 & 2 \\ 3 & 0.5 \\ -0.1 & -4 \end{bmatrix} \quad {\vec{\mathbf{b}}}^{[1]} = \begin{bmatrix} 1 & -2 & 0.3 \end{bmatrix}$$

We calculated the gradients of the cost function with respect to the parameters of layer 1:

$$
\frac{\partial J(\theta)}{\partial {W^{[1]}}} = \begin{bmatrix} -0.0249 & -0.0396 \\ -0.1814 & -0.2428 \\ 0 & 0 \end{bmatrix}$$
And

$$\frac{\partial J(\theta)}{\partial {\vec{\mathbf{b}}}^{[1]}} = \begin{bmatrix} -0.0147 & -0.0614 & 0 \end{bmatrix}
$$


**Update Parameter ${\vec{\mathbf{b}}}^{[1]}$:**
$$
{\vec{\mathbf{b}}}^{[1]} = {\vec{\mathbf{b}}}^{[1]} - \alpha \cdot \frac{\partial J(\theta)}{\partial {\vec{\mathbf{b}}}^{[1]}}
$$

We defined our learning rate $\alpha = 0.01$.

So, we have:

$$
{\vec{\mathbf{b}}}^{[1]} = \begin{bmatrix} 1 & -2 & 0.3 \end{bmatrix} - 0.01 \cdot \begin{bmatrix} -0.0147 & -0.0614 & 0 \end{bmatrix}
$$

Which is equal to:

$$
{\vec{\mathbf{b}}}^{[1]}_{\text{updated}} = \begin{bmatrix} 0.1001 & -1.9994 & 0.3 \end{bmatrix}
$$

**Update Parameter $W^{[1]}$:**

$$
W^{[1]} = W^{[1]} - \alpha \cdot \frac{\partial J(\theta)}{\partial {W^{[1]}}}
$$

So, we have:

$$
W^{[1]} = \begin{bmatrix} -1 & 2 \\ 3 & 0.5 \\ -0.1 & -4 \end{bmatrix} - 0.01 \cdot \begin{bmatrix} -0.0249 & -0.0396 \\ -0.1814 & -0.2428 \\ 0 & 0 \end{bmatrix}
$$

Which is equal to:

$$
W^{[1]}_{\text{updated}} = \begin{bmatrix} -0.9998 & 2.0004 \\ 3.0018 & 0.5024 \\ -0.1 & -4 \end{bmatrix}
$$

Let's verify our calculation with PyTorch.

When using PyTorch, after calling `backward()` on the cost, which runs the backpropagation algorithm and calculate gradients of the cost function with respect to all the parameters of the neural network, we can update the parameters using the `optimizer.step()` method.

We defined our optimizer as `optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`. This means we are using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01.


```python
optimizer.step()
print_model_parameters(model)
```

    Layer 1: Linear

    weight: torch.Size([3, 2]) Parameter containing:
    tensor([[-0.9998,  2.0004],
            [ 3.0018,  0.5024],
            [-0.1000, -4.0000]], requires_grad=True)
    weight.grad:
    tensor([[-0.0249, -0.0396],
            [-0.1814, -0.2428],
            [ 0.0000,  0.0000]])

    bias: torch.Size([3]) Parameter containing:
    tensor([ 1.0001, -1.9994,  0.3000], requires_grad=True)
    bias.grad:
    tensor([-0.0147, -0.0614,  0.0000])
    --------------------------------------------------
    Layer 2: Linear

    weight: torch.Size([2, 3]) Parameter containing:
    tensor([[ 0.5038,  1.0057, -2.0000],
            [ 0.6982,  0.0968,  0.3000]], requires_grad=True)
    weight.grad:
    tensor([[-0.3831, -0.5747,  0.0000],
            [ 0.1752,  0.3175,  0.0000]])

    bias: torch.Size([2]) Parameter containing:
    tensor([-3.9994,  4.9998], requires_grad=True)
    bias.grad:
    tensor([-0.0639,  0.0246])
    --------------------------------------------------
    Layer 3: Linear

    weight: torch.Size([1, 2]) Parameter containing:
    tensor([[ 0.5102, -0.2907]], requires_grad=True)
    weight.grad:
    tensor([[-1.0216, -0.9253]])

    bias: torch.Size([1]) Parameter containing:
    tensor([0.1008], requires_grad=True)
    bias.grad:
    tensor([-0.0821])
    --------------------------------------------------
