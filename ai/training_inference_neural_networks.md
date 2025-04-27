---
date: "2025-04-02"
draft: false
title: "Training and Inference in Neural Networks"
description: "An overview of complete cycle of training and inference in neural networks."
tags:
    - "AI"
    - "Neural Networks"
---

In this section, we'll discuss the cycle of training and inference in neural networks. We'll assume that we already have a preprocessed dataset which has passed through the
[Exploratory Data Analysis (EDA) and Feature Engineering](ai_machine_learning_overview.md#exploratory-data-analysis-eda-and-feature-engineering) stages and our dataset is now ready for training.


## Training
Training a neural network is the process of optimizing the model's parameters (weights and biases) to minimize the [cost function](loss_cost_functions_machine_learning.md). The most common optimization algorithm used for training neural networks is [Gradient Descent and its Variants](gradient_descent_machine_learning.md), in particular the [mini-batch Stochastic Gradient Descent (SGD)](gradient_descent_machine_learning.md#mini-batch-sgd).

Let's say we have $m=60,000$ training examples, and we decide to use mini-batch SGD with a batch size of $size=64$, then we have:
```sh
for each epoch:
    batches = split dataset into random batches of 64 examples
    for batch in batches:
        # Run all steps for each batch of 64 examples (one iteration)
        forward propagation # Calculate predictions
        compute cost # Compare predictions with actual values
        backward backpropagation # Compute the gradients
        update all parameters # Reduce the cost
```
[**Epoch**](gradient_descent_machine_learning.md#epochs-and-iterations)<br>
One complete pass through the entire training dataset.

$$\text{one epoch} = \text{processing 60,000 training examples}$$

[**Iteration**](gradient_descent_machine_learning.md#epochs-and-iterations)<br>
One complete pass through a mini-batch of training examples. In our example:

$$\text{one iteration} = \text{processing 64 training examples}$$

$$ \text{number of batches(iterations)} = \frac{m}{size} = \frac{60,000}{64} = 937.5 \approx 938$$


**Repeat Until Convergence**:<br>
The number of epochs is a hyperparameter that you can set. It defines how many times we go through the entire training dataset and update the model parameters. This cycle of training is repeated until the model converges, meaning that the cost function reaches a satisfactory level or stops reducing significantly.

**Computational Graph**<br>
Computation graph is break down of the neural network (forward propagation and back propagation) into smaller building blocks, which then can be calculated step by step. This graph is a directed acyclic graph (DAG), and output of each node is calculated based on the output of the previous nodes. So, it can reuse the output of the previous nodes to calculate the output of the next nodes, which makes the computation more efficient.

See [Computational Graph](computational_graph_machine_learning.md) for more details.

![](https://pooya.io/ai/images/nn_computational_graph.svg)


### Forward Propagation
In forward propagation we calculate the predictions of the neural network based on the **current** parameters (weights and biases) and the input data.

We move forward (Left to Right) from the input node to the output node.

See [Forward Propagation](forward_propagation_neural_networks.md) for more details.

### Compute Cost/Loss
In [Compute cost/loss](loss_cost_functions_machine_learning.md) step, we compare the predictions with the actual values (ground truth) and calculate the cost (loss) of the predictions.


### Backpropagation
In backpropagation, we calculate the gradients of the cost function with respect to the parameters (weights and biases) using the backpropagation algorithm.

Backpropagation is an algorithm to calculate the [derivative (gradient)](../math/derivatives.md) of the cost function with respect to the parameters of the neural network.

The name of the Backpropagation which is also called _back prop_ or _backward pass_, coming from the fact that after the [forward pass](neural_networks_inference.md) which calculates the output of the network based on the current parameters, then backprop calculates the derivative (gradient) of the cost function with respect to the parameters in the reverse order of the forward pass, meaning from the output layer back to the first layer. Hence, the name back propagation.



In backpropagation, we use the computational graph and move backwards (Right to Left), from the loss node to the input node and calculate the derivative of the cost function with respect to each parameter in the neural network.


In case of backpropagation, the computation graph is like breaking down the chain rule of the derivative into each composite function, and then calculate from the most outer function (loss function) to the most inner function (the parameters of the neural network).

Without the computational graph, we have to calculate the derivative of the loss function with respect to each parameter one by one which is very inefficient. However, with the computational graph, we start from the loss function (end of the graph) and calculate the derivative of the loss function with respect to the output of each function, then keep going backward to calculate the derivative of the loss function with respect to each parameter in the neural network. In this way, we reuse the output of the previous derivatives to calculate the next derivatives, which makes the computation more efficient.

See [Backpropagation Example](backpropagation_neural_networks.md) for see how it works in action.


### Update Parameters
So far, all the calculations are based on the **current** parameters. In this step, based on the calculated gradients from the backpropagation step, we update the parameters to move in the direction of the steepest descent (negative gradient) to reduce the cost function.

See [here](https://pooya.io/ai/gradient-descent-machine-learning/#4-update-the-parameters) for more details.

## Inference
Inference is the process of making predictions using the trained neural network (optimized parameters). Inference is the same as [Forward Propagation](forward_propagation_neural_networks.md) process. That's why terms Inference, Forward Propagation, Forward Pass, and Prediction are often used interchangeably.



## Example
Let's look at all these steps of training and inference in a complete example. We will assume that our data is already preprocessed and is ready for training. We will show a simple but practical example using PyTorch.

We will use the MNIST dataset, which consists of handwritten digits from 0 to 9. We will build a neural network to classify these digits. So, this is a _classification_ problem with 10 classes (digits 0 to 9).

In this example, we keep the model and steps simple. For a more advanced implementation, see [this official PyTorch example](https://github.com/pytorch/examples/blob/main/mnist/main.py).



### Training
#### Data
As always in machine learning, we start with exploring our data and [Exploratory Data Analysis (EDA)](https://pooya.io/ai/ai_machine_learning_overview/#exploratory-data-analysis-eda-and-feature-engineering)

The MNIST dataset contains 60,000 training images and 10,000 test images.
- Each image is 28x28 pixels, and the labels are the digits from 0 to 9.
- The images are grayscale (one channel), so the pixel values range from 0 to 255, which is the brightness (intensity) of the pixel.
- The dataset is split into a training set and a test set.

The training data set is a matrix of 60,000 rows and 28x28x1= 784 columns. Each row represents a single image, which has 784 columns (features).

$$Channel \times Height \times Width = 1 \times 28 \times 28 = 784$$




$$X_{\text{train}} \in \mathbb{R}^{60000 \times 784}$$

$$X_{\text{train}} = \begin{bmatrix}
\vec{\mathbf{x}}^{(1)} \\
\vec{\mathbf{x}}^{(2)} \\
\vdots \\
\vec{\mathbf{x}}^{(60000)}
\end{bmatrix}$$

Where:
- $\vec{\mathbf{x}}^{(i)} \in \mathbb{R}^{784}$ is the $i$-th image in the training set.

$X_{\text{test}}$ similarly is a matrix of 10,000 rows and 784 columns.



The labels are a vector of same size (60,000) as the number of training images. Each label is an integer from 0 to 9, representing the digit in the corresponding image.

$$y_{\text{train}} \in \mathbb{R}^{60000}$$

Similarly, $y_{\text{test}}$ is a vector of size 10,000.

$$y_{\text{test}} \in \mathbb{R}^{10000}$$


```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
```

Let's download the MNIST dataset using `torchvision` offered by PyTorch.


```python
# Define the transformation to be applied to the images
transform = transforms.Compose([transforms.ToTensor()])

# Download the MNIST training and test datasets
train_data = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)
```


```python
print(
    f"X_train shape: {train_data.data.shape}, dtype: {train_data.data.dtype}"
)
print(f"y_train shape: {train_data.targets.shape}")

print(f"X_test shape: {test_data.data.shape}, dtype: {test_data.data.dtype}")
print(f"y_test shape: {test_data.targets.shape}")
```

    X_train shape: torch.Size([60000, 28, 28]), dtype: torch.uint8
    y_train shape: torch.Size([60000])
    X_test shape: torch.Size([10000, 28, 28]), dtype: torch.uint8
    y_test shape: torch.Size([10000])



```python
print(f"First image shape: {train_data.data[0].shape}")
print(f"First image label: {train_data.targets[0]}")
```

    First image shape: torch.Size([28, 28])
    First image label: 5


**Labels:**

In MNIST dataset, where we are classifying handwritten digits from 0 to 9, the labels are simply the digits themselves. In other words, the class 0 (label 0) corresponds to the digit 0, class 1 (label 1) corresponds to the digit 1, and so on.

| Class (Label) | Value |
|---------------|-------|
| 0             | 0     |
| 1             | 1     |
| 2             | 2     |
| 3             | 3     |
| 4             | 4     |
| 5             | 5     |
| 6             | 6     |
| 7             | 7     |
| 8             | 8     |
| 9             | 9     |

However, in a more complex dataset, the labels are not always integers. For example, in [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file#labels), the labels are strings representing the class names. The labels are as follows:

| Class (Label) | Value |
|---------------|-------|
| 0             | T-shirt/top |
| 1             | Trouser     |
| 2             | Pullover    |
| ...           | ...         |

So, this is important to note that regardless of the actual value of the classes, we always map them to integers starting from 0 which in that case the logits of the output layer are automatically mapped to the corresponding index of the class. For example, $z_{0}$ will be mapped to class 0, $z_{1}$ will be mapped to class 1, and so on.

**Batching:**

As we discussed [here](https://pooya.io/ai/gradient-descent-machine-learning/#types-of-gradient-descent), one of the most commonly techniques in training a neural network is to use [mini-batch SGD (Stochastic Gradient Descent)](https://pooya.io/ai/gradient-descent-machine-learning/#mini-batch-sgd). In mini-batch SGD, we divide the training dataset into smaller chunks (batches) and go through all steps of forward and backward propagation for each batch.

In PyTorch, we can do this by wrapping our dataset in a `DataLoader` object which allows us to iterate over the dataset in batches and also support for shuffling, sampling, and multi-processing. The `DataLoader` object is the one that feeds the data to the model batch by batch.

We define batch size as 64. It means in each iteraction of the training (Gradient Descent) we will use 64 images to calculate the cost, and gradients and then update the parameters of the model.


```python
batch_size = 64

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
```

As we saw earlier our training data is 3D matrix of size (60000, 28, 28). In other words, we have 60,000 examples which each is a 28x28 pixels image. So, overall we have a matrix of 60,000 rows which each row is a matrix of 28 rows and 28 columns.

However, as soon as we wrap it in the `DataLoader` object, then it breaks down the whole dataset into batch size chunks.


```python
for X, y in train_dataloader:
    print(
        f"Shape of X [N=Batch size, C=Channel, H=Height, W=Weight]: {X.shape}"
    )
    print((torch.flatten(X, start_dim=0)).shape)
    print((torch.flatten(X, start_dim=1)).shape)

    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```

    Shape of X [N=Batch size, C=Channel, H=Height, W=Weight]: torch.Size([64, 1, 28, 28])
    torch.Size([50176])
    torch.Size([64, 784])
    Shape of y: torch.Size([64]) torch.int64


The shape of $X$ is (N=batch size, C=channel, H=height, W=width) which is (64, 1, 28, 28) in our case. The first dimension is the batch size, the second dimension is the channel (1 for grayscale). This channel dimension arrangement comes from the `transforms.ToTensor()` which we added during data loading. While PIL (Python Imaging Library) typically represents images with channels as the last dimension (height, width, channels) or without explicit channels for grayscale, PyTorch's convention is to have channels first (NCHW format). The `ToTensor()` transform handles this conversion, moving from PIL's format to PyTorch's expected tensor format of (channels, height, width), and then the DataLoader adds the batch dimension.

**Flattening**:<br>
In this example, for simplicity we are going to use a fully connected (Dense) layers as the input layer, which expects an input $X$ of 2D matrix with shape of  $(m \times n)$ where $m$ is the number of examples (batch size) and $n$ is the number of features (784 in our case). So, we need to flatten the input dataset to be a 2D matrix of size (rows = batch size, columns = features of each image).

> Dense layers expect a 2‑D input of shape $(\text{batch\_size},\ \text{num\_features})$, so images must be flattened to vectors first. But other layers such as convolutional layers, may not need this flattening.


- **Dense (fully‑connected) layer** – Every sample must come in as one long row‑vector of features, so for a batch you need a 2‑D array of shape:
    $$(m,\; n)= (\text{batch\_size},\; \text{features})$$
    For an MNIST image that’s $28\times28=784$ features.

- **Convolutional layer** – Convolutional layers comes in different shapes (e.g. `Conv1d`, `Conv2d`, etc). A 2‑D convolution over images expects **four** axes $(N,\;C,\;H,\;W)$.

So, we use PyTorch [flatten](https://pytorch.org/docs/stable/generated/torch.flatten.html) function to flatten the input dataset. We use `torch.flatten()` with `start_dim=1` to flatten the input starting from the second dimension onwards. This means we want to keep the first dimension (batch size) as is, and flatten the rest of the dimensions (channel, height, width) into a single dimension, $1 \times 28 \times 28 = 784$. So, the result will be a 2D tensor of size $(64, 784)$.


#### Creating the Model

In this example we create a very simple model using 3 fully connected layers (also called linear layers, or Dense layers).

**Layer 1 (Dense):**
- $28 \times 28 \times 1 = 784$ inputs, $512$ outputs
- Activation function: ReLU
- In this layer we have $512$ neurons. the shape of matrix $W1$ is (512, 784) and the shape of vector $b1$ is (512,)

**Layer 2 (Dense):**
- $512$ inputs, $512$ outputs.
- Activation function: ReLU
- In this layer we have $512$ neurons. the shape of matrix $W2$ is (512, 512) and the shape of vector $b2$ is (512,)

**Layer 3 (Dense):**
- $512$ inputs, $10$ outputs (one for each class)
- Activation function: None (linear Activation)
- In this layer we have $10$ neurons. the shape of matrix $W3$ is (10, 512) and the shape of vector $b3$ is (10,)

**Placement of Activation Function for the Output Layer:**<br>

As we discussed [here](https://pooya.io/ai/forward-propagation-neural-networks/#output-layer-logits-and-activation-function-placement), the output layer's activation function is applied separately to the logits of the output layer. In here, we have a multi-class classification problem, so we will use the softmax activation function.




```python
# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(in_features=28 * 28, out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Flatten the input tensor
        x = torch.flatten(x, start_dim=1)

        # Layer 1
        z1 = self.linear1(x)
        a1 = F.relu(z1)

        # Layer 2
        z2 = self.linear2(a1)
        a2 = F.relu(z2)

        # Layer 3
        # z3 are the logits
        logits = self.linear3(a2)
        return logits
```

Before we initialize the model, we want to find out if on the current machine (which we about to execute the computations of our model) we have a GPU or not. If we have GPU, we almost always prefer to use it over CPU for training and inference computations.



```python
# pick accelerator device (GPU, MPS, etc) if available, otherwise CPU.
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print(f"Using {device} device")
```

    Using cpu device


Now let's initialize our model. If we don't have GPU, or if we didn't call `.to(device)` on the model, then it falls back to CPU.



```python
# Initialize the model
model = NeuralNetwork().to(device)


def print_model_summary(layer):
    for name, layer in model.named_children():
        print(f"{name}:")
        print(f"\tLayer type: {layer}")
        print(f"\tOutput shape (number of neurons): {layer.out_features}")
        print(f"\tWeights shape: {layer.weight.shape}")
        print(f"\tBias shape: {layer.bias.shape}")
        print(
            f"\tTotal parameters: {layer.weight.numel() + layer.bias.numel()}"
        )
    print(
        f"Model Total parameters: {sum(p.numel() for p in model.parameters())}"
    )


print_model_summary(model)
```

    linear1:
	Layer type: Linear(in_features=784, out_features=512, bias=True)
	Output shape (number of neurons): 512
	Weights shape: torch.Size([512, 784])
	Bias shape: torch.Size([512])
	Total parameters: 401920
    linear2:
	Layer type: Linear(in_features=512, out_features=512, bias=True)
	Output shape (number of neurons): 512
	Weights shape: torch.Size([512, 512])
	Bias shape: torch.Size([512])
	Total parameters: 262656
    linear3:
	Layer type: Linear(in_features=512, out_features=10, bias=True)
	Output shape (number of neurons): 10
	Weights shape: torch.Size([10, 512])
	Bias shape: torch.Size([10])
	Total parameters: 5130
    Model Total parameters: 669706


The model and number of parameters are as expected. For example:

For Layer 1:
- Weights: $512 \times 784 = 401408$
- Biases: $512$
- Total Parameters: $401408 + 512 = 401920$

And so on for the other layers.

**Softmax Placement:**<br>
As we discussed earlier, for numerical stability, the output layer of our model should be a _linear_ layer (no activation function) which means the output of the last layer is the logits $z$. Then we place our softmax functions outside of the model and use it depending on if we are in _training_ or _inference_ mode.
- At the time of **inference**, we apply the softmax activation function (or simply argmax) to the logits to get the highest probability class.
- At the time of **training**, we pass the logits to the Cross-Entropy Loss function directly. See [here](https://pooya.io/ai/loss-cost-functions-machine-learning/#sparse-categorical-cross-entropy-loss) for more details.

> Note: Since the model's output is logits, the output of the model is **not** probabilities. To get probabilities, we need to apply the softmax function to the logits.

#### Training the Model

**Define Model Optimization Algorithm**

We will use the [mini-batch SGD (Stochastic Gradient Descent)](https://pooya.io/ai/gradient-descent-machine-learning/) optimizer with learning rate of `0.001`.



```python
# Stochastic Gradient Descent (SGD) optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```

**Loading Training Data to the Device:**<br>
In previous steps, we checked if we have any hardware accelerators (GPU, MPS, etc) available or not. We almost always perfer using hardware accelerators such as (GPU, MPS, etc) for machine learning computations over CPU if available.

Then in PyTorch, we can use the `.to(device)` method on the model to move the model's parameters to that device and run the model's computations on the same device. Now, we also want to move the training data to the same device as well.

When we say, _move the data to the device_, we mean we want the data be on the hardware memory which is local to the choosen processor (device). For example, if we are using a GPU, we want the data to be in the GPU's memory (onboard RAM) instead of the CPU's memory (main RAM).

1. **CPU tensors**
   - Live in the system’s main RAM.
   - All arithmetic on them is done by the CPU cores.

2. **GPU tensors**
   - Live in the GPU’s dedicated onboard RAM (VRAM) which is physically separate from the CPU’s RAM.
   - All arithmetic on them is done by the GPU’s streaming multiprocessors.


Note that if we don't move the data to the local memory of the device, and model and model's parameters are on the GPU, and the training data is on the CPU, then moving data between CPU and GPU involves transferring data over the PCI-Express bus, which is relatively slow compared to accessing memory local to each processor


```python
def train(model, device, train_loader, optimizer, dry_run=False):
    # Set the model to the training mode
    model.train()

    # Go through the training data, batch by batch
    for batch_idx, (X, y) in enumerate(train_loader):

        # Move the training batch to the device where the model is.
        X, y = X.to(device), y.to(device)

        # Reset any previous calculated gradients to zero
        optimizer.zero_grad()

        # Forward propagation (inference)
        logits = model(X)

        # Calculate the cost
        loss = F.cross_entropy(logits, y)

        # Backpropagation (calculate gradient)
        loss.backward()

        # Update all model's parameters (using the calculated gradients)
        optimizer.step()

        # Print the progress log every 100 batches
        if batch_idx % 100 == 0:
            batch = batch_idx + 1 if batch_idx == 0 else batch_idx
            processed = batch * len(X)
            size = len(train_loader.dataset)
            progress = 100.0 * batch / len(train_loader)

            print(
                f"[Batch:{batch:>5d}/{len(train_loader)} "
                f"Processed: {processed:>5d}/{size} "
                f"({progress:.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )
            if dry_run:
                break
```

Let's explain some of the details of the `train()` function:

`traing()` has 4 input parameters:
- `dataloader`: To feed the training data to the model in batches.
- `model`: Our neural network model which are going to train.
- `optimizer`: The optimization algorithm (i.e variation of SGD).
- `dry_run`: A boolean flag to indicate if we are in dry run mode or not. This is specially useful for debugging and troubleshooting when we want to test the training loop without actually going through all whole training data.



```python
model.train()
```
This line sets the model to training mode. This is important because some layers (like dropout and batch normalization) behave differently during training and inference. For example, in training mode, dropout layers will randomly drop some neurons, while in inference mode, they will use all neurons.


**Iteration:**<br>
We now iterate over the training data using the `dataloader` object batch by batch.

```python
for batch_idx, (X, y) in enumerate(train_loader):
    ...
```

Each round of the loop, is the **complete** process of training (forward and backward propagation, and parameters update) for a **single batch** from the training data. This is called **one iteration**. See [here](https://pooya.io/ai/gradient-descent-machine-learning/#gradient-descent-summary) for more details.

In here, we only need `X` and `y` which are in the size of (64, 784) and (64,) respectively. We also get the `batch_idx` which is the index of the current batch for printing the progress of the training.

> Note: All the steps of forward propagation, loss calculation, backward propagation, and parameters update are done **only** for the current batch of data (which is this example is 64 sample images)


**Loss Function:**<br>
In PyTorch, the word **loss** is often used to refer to the **cost** function. Although [Loss and Cost functions](https://pooya.io/ai/loss-cost-functions-machine-learning/) are not the same, in PyTorch term **loss** is used to refer to both.

We will use [Cross-Entropy Loss](https://pooya.io/ai/loss-cost-functions-machine-learning/#cross-entropy-loss) function as we are dealing with a multi-class classification problem.

```py
loss = F.cross_entropy(logits, y)
```
[`cross_entropy()`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) in combines the softmax activation function and the negative log-likelihood loss function in one single function. It expects the logits as input and applies the softmax function internally.

Let's run the training for a few epochs and see the results.



```python
epochs = 3
for t in range(epochs):
    print(f"Epoch {t + 1}/{epochs}\n-------------------------------")
    train(model, device, train_dataloader, optimizer)
    print()
```

    Epoch 1/3
    -------------------------------
    [Batch:    1/938 Processed:    64/60000 (0%)]	Loss: 2.303745
    [Batch:  100/938 Processed:  6400/60000 (11%)]	Loss: 2.298823
    [Batch:  200/938 Processed: 12800/60000 (21%)]	Loss: 2.300962
    [Batch:  300/938 Processed: 19200/60000 (32%)]	Loss: 2.277889
    [Batch:  400/938 Processed: 25600/60000 (43%)]	Loss: 2.285777
    [Batch:  500/938 Processed: 32000/60000 (53%)]	Loss: 2.286932
    [Batch:  600/938 Processed: 38400/60000 (64%)]	Loss: 2.279493
    [Batch:  700/938 Processed: 44800/60000 (75%)]	Loss: 2.274683
    [Batch:  800/938 Processed: 51200/60000 (85%)]	Loss: 2.266902
    [Batch:  900/938 Processed: 57600/60000 (96%)]	Loss: 2.260534

    Epoch 2/3
    -------------------------------
    [Batch:    1/938 Processed:    64/60000 (0%)]	Loss: 2.258941
    [Batch:  100/938 Processed:  6400/60000 (11%)]	Loss: 2.250776
    [Batch:  200/938 Processed: 12800/60000 (21%)]	Loss: 2.261983
    [Batch:  300/938 Processed: 19200/60000 (32%)]	Loss: 2.223863
    [Batch:  400/938 Processed: 25600/60000 (43%)]	Loss: 2.239783
    [Batch:  500/938 Processed: 32000/60000 (53%)]	Loss: 2.239365
    [Batch:  600/938 Processed: 38400/60000 (64%)]	Loss: 2.221487
    [Batch:  700/938 Processed: 44800/60000 (75%)]	Loss: 2.231184
    [Batch:  800/938 Processed: 51200/60000 (85%)]	Loss: 2.209245
    [Batch:  900/938 Processed: 57600/60000 (96%)]	Loss: 2.198760

    Epoch 3/3
    -------------------------------
    [Batch:    1/938 Processed:    64/60000 (0%)]	Loss: 2.195798
    [Batch:  100/938 Processed:  6400/60000 (11%)]	Loss: 2.180521
    [Batch:  200/938 Processed: 12800/60000 (21%)]	Loss: 2.205530
    [Batch:  300/938 Processed: 19200/60000 (32%)]	Loss: 2.140531
    [Batch:  400/938 Processed: 25600/60000 (43%)]	Loss: 2.168901
    [Batch:  500/938 Processed: 32000/60000 (53%)]	Loss: 2.162959
    [Batch:  600/938 Processed: 38400/60000 (64%)]	Loss: 2.126863
    [Batch:  700/938 Processed: 44800/60000 (75%)]	Loss: 2.154950
    [Batch:  800/938 Processed: 51200/60000 (85%)]	Loss: 2.111911
    [Batch:  900/938 Processed: 57600/60000 (96%)]	Loss: 2.093282



The following above output clearly shows how **Iteration** and **Epochs** are different from each other.
- **Iteration**: One complete cycle of training (forward and backward propagation, and parameters update) for a _single batch_ of data.
- **Epoch**: One complete cycle of training (forward and backward propagation, and parameters update) for _all batches_ (entire training dataset).

As you can see in each iteration, we process **one batch** of data (64 images in this example). The process includes **all** the training steps (forward, cost, backward, and update), but **only** for that **one batch** of data.

When we go through all the batches, we have processed the entire training dataset. This is called **one epoch**.

In below, we have 60,000 training images, and batch size of 64.

$$\text{Number of Batches} = \frac{\text{Number of Training Images}}{\text{Batch Size}} = \frac{60000}{64} = 937.5 \approx 938
$$

```
Epoch 1/3
-------------------------------
[Batch:    1/938 Processed:    64/60000 (0%)]	Loss: 2.305700
[Batch:  100/938 Processed:  6400/60000 (11%)]	Loss: 2.299334
...
[Batch:  900/938 Processed: 57600/60000 (96%)]	Loss: 2.252129
```

For brevity, we only print the progress of the training every 100 iterations (batches). For example, the second row shows that so far, we have processed 100 batches (each batch is 64 images) which means we have processed $100 \times 64 = 6400$ images, and the loss after processing these 100 batches is 2.299334.


#### Evaluation of the Model
[Evaluation](https://pooya.io/ai/model-evaluation-machine-learning/) is the key step in the training cycle of a machine learning models. Evaluation helps us to understand and how well our model is in [generalization](https://pooya.io/ai/generalization-machine-learning/) which is the ability of the model to perform well on unseen data. In other words, we want to see how well our model is able to generalize to new data that it has not seen before and how to improve it if needed.

One of the steps in evaluation is to [split the dataset into training and test datasets](https://pooya.io/ai/model-evaluation-machine-learning/). In this example, we are using MNIST dataset which comes with split of training and test datasets. $60,000$ training images and $10,000$ test images.



```python
print(f"X_test shape: {test_data.data.shape}, dtype: {test_data.data.dtype}")
print(f"y_test shape: {test_data.targets.shape}")
print(len(test_dataloader.dataset))
```

    X_test shape: torch.Size([10000, 28, 28]), dtype: torch.uint8
    y_test shape: torch.Size([10000])
    10000



```python
def test(model, device, test_loader):
    # Set the model to the evaluation mode
    model.eval()

    # Initialize the test loss and correct predictions
    test_loss = 0
    correct = 0

    # We don't need to calculate gradients for the test data, so we use
    # torch.no_grad() to save memory and computations
    with torch.no_grad():

        # Go through the test data, batch by batch
        for X, y in test_loader:

            # Move the test batch to the device where the model is.
            X, y = X.to(device), y.to(device)

            # Forward propagation (inference)
            logits = model(X)

            # Calculate the cost for each batch and sum it up
            test_loss += F.cross_entropy(logits, y, reduction="sum").item()

            # Get the predicted class
            pred = logits.argmax(dim=1, keepdim=True)

            # Count the number of correct predictions
            correct += pred.eq(y.view_as(pred)).sum().item()

    # Calculate the average loss across all batches
    test_loss /= len(test_loader.dataset)

    # Print the test results
    print(
        f"Test Result:\n Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100.0 * correct / len(test_loader.dataset):.0f}%)"
    )
```

Let's decompose the `test()` function and execute it line by line.


The function `test()` has input parameters:
- `model`: Our model which has been trained and we want to evaluate.
- `device`: The device (e.g. GPU, CPU, etc) on which the model is running.
- `test_loader`: The test dataloader which feed the **test** dataset to the model in batches.

We first set the model to evaluation mode, and initialize our counters.


```python
model.eval()

# Initialize the test loss and correct predictions
test_loss = 0
correct = 0
```


This line sets the model to evaluation mode. This is important because some layers (like dropout and batch normalization) behave differently during training and inference.

Note: even though the name of this function is `eval()`, it will be used for both evaluation and inference. In PyTorch, the `eval()` function simply sets the model for final inference mode.


```python
torch.no_grad()
```
As we discussed in the [Computational Graph](https://pooya.io/ai/computational-graph-machine-learning/) and `autograd`, PyTorch regardless of the mode (training or inference) creates a computational graph, stores the intermediate activations for backpropagation, and keeps track of the gradients. During the training, we want this behavior as we need to calculate the gradients for the backpropagation, right after the forward pass. But during pure inference, we don't need to build the computational graph and store any values. So, we use `torch.no_grad()` tells PyTorch to skip building the computational graph, which saves memory and speeds up the computation.

> In line-by-line execution for learning purposes, we can omit this line for now, but in practice, we should always use it during inference to save memory and speed up the computation.


```python
# Grab our first batch of test data
X, y = next(iter(test_dataloader))
print(f"X shape: {X.shape}, y shape: {y.shape}")
```

    X shape: torch.Size([64, 1, 28, 28]), y shape: torch.Size([64])



```python
# Move the test batch to the device where the model is.
X, y = X.to(device), y.to(device)

# Forward propagation (inference)
logits = model(X)
print(f"Logits shape: {logits.shape}")
print(f"First 2 logits:\n{logits[0:2]}")
```

    Logits shape: torch.Size([64, 10])
    First 2 logits:
    tensor([[-0.0442, -0.0884, -0.0270, -0.0364, -0.0347, -0.1009, -0.1359,  0.2878,
              0.0304,  0.1313],
            [ 0.2418, -0.1018,  0.1556,  0.0864, -0.1910,  0.0196,  0.0774, -0.2082,
              0.0541, -0.1250]], grad_fn=<SliceBackward0>)



```python
batch_loss_sum = F.cross_entropy(logits, y, reduction="sum").item()
print(f"Sum of batch loss: {batch_loss_sum:.4f}")

test_loss += batch_loss_sum
```

    Sum of batch loss: 135.4241


This above calculates the losses for the entire batch and **adds** them up together in `test_loss` variable. We broke it down for better understanding.

Previously we used `F.cross_entropy(logits, y)` in training which returns the **average** loss for the batch. The default value for `reduction="mean"` which calculates the average loss for the batch.

$$J = \frac{1}{m} \sum_{i=1}^{m} L(y^{(i)}, \hat{y}^{(i)})$$

Where:
- $m$ is the number of examples in the batch (64 in this example).

However, since the last batch is usually smaller than other batches, the easiest way to calculate the average of losses across the entire test dataset, is to simply add all losses together and then divide the total by the number of examples in the test dataset.

```python
test_loss /= len(test_loader.dataset)
```

Note: During the training the `loss` is the final node of the computational graph, which we want walk backwards and calculate the gradients (backpropagation) from. But during the inference, we don't need to do that or keeping the graph. So, the `item()` function is simply get the value of the loss tensor (which is a matrix of size 1x1) and convert it to a Python float.






```python
pred = logits.argmax(dim=1, keepdim=True)

print(f"pred shape: {pred.shape}")
print(f"First 2 predictions:\n{pred[0:2]}")
```

    pred shape: torch.Size([64, 1])
    First 2 predictions:
    tensor([[7],
            [0]])


`pred` is the index of **most probable class** for each sample in the batch. We have 64 samples in the batch, so `pred` is a vector of size (64,1), one predicted class for each sample.

As we discussed in [Forward Propagation](https://pooya.io/ai/forward-propagation-neural-networks/), the output of the modelis the logits (not the probabilities). This is a matrix of size (batch size, number of classes). In this example, the logits are a matrix of size (64, 10) which means we have 64 samples and 10 classes.

$$\text{logits} = Z^{(3)} = \begin{bmatrix}
z_{1}^{(1)} & z_{2}^{(1)} & \cdots & z_{10}^{(1)} \\
z_{1}^{(2)} & z_{2}^{(2)} & \cdots & z_{10}^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
z_{1}^{(64)} & z_{2}^{(64)} & \cdots & z_{10}^{(64)}
\end{bmatrix}$$

Where:
- $z_{i}^{(j)}$ is the $i$-th logit for the $j$-th sample. So, the first row is the logits for the first sample.
- Each logit is a real number which is the output of linear transformation of the last layer.

In order to get the probabilities, we need to apply the softmax function to the logits.

$$\text{probabilities} = A^{(3)} = \text{softmax}(Z^{(3)})$$

As we discussed [here](https://pooya.io/ai/classification-neural-networks/), softmax for logits is defined as:

$$\text{softmax}(z_{i}) = \frac{e^{z_{i}}}{\sum_{j=1}^{N} e^{z_{j}}}$$

Where:
- $z_{i}$ is the $i$-th logit.
- $N=10$ is the number of classes (digits 0 to 9).



If we look into how the softmax function works, we can see that if our ultimate goal is not getting the probabilities, but finding the **most likely class** (the predicted class), we can simply pick the largest output from the logits. In that case, the softmax is not required and we can use a simpler operation called `argmax` which simply returns the index of the largest value in the given vector.

Since the logits are a matrix of size (64, 10), we want to apply the `argmax` function on each row (10 logits of each sample) to get the index of largest value (most likely class) for that row. So, we set `dim=1` to apply the `argmax` function on the second dimension (columns) of the logits matrix. The result will be a vector of size (64,). In order to keep the result in the same shape as the logits, we set `keepdim=True` which then returns a matrix of size (64, 1).

**Compare the Predicted Classes with the Target Labels:**<br>



```python
# Compare the predicted class with the actual class
for i in range(2):
    print(
        f"Example {i}: Predicted class: {pred[i].item()}"
        f", Actual class: {y[i].item()}"
    )
```

    Example 0: Predicted class: 7, Actual class: 7
    Example 1: Predicted class: 0, Actual class: 2



`y` is the vector of target labels (ground truth) for the current batch. The shape of `y` is (64,). We want to compare the predicted classes (pred) with the target labels (y) element-wise. `pred` is what we calculated using `argmax` function in the previous step. The shape of `pred` is (64, 1) because we set `keepdim=True`, so, we need to reshape `y` to be the same shape as `pred` in order to compare them element-wise. We can do this by using `view_as(pred)` which reshapes `y` to the same shape as `pred`.

- `sum()` in PyTorch sums all the `True` values as 1 and `False` values as 0. So, the result will be a matrix of size (64, 1) with 1s and 0s.

Then we use the `eq()` function to compare the two tensors element-wise. The result will be a matrix of size (64, 1) with 1s and 0s. Finally, we use `sum()` to count the number of correct predictions and convert it to a Python integer using `item()`.


>Note: It's not strictly necessary here to first keep the shape of output using `keepdim=True` and then use `view_as(pred)` to reshape the `y` tensor to the same shape as `pred`. We could have let the `pred` tensor be a 1-D tensor of size (64,) and then used `y` directly in the comparison. However, this is a common practice in to keep the tensor ranks consistent throughout the pipeline of training and inference (which could also be useful for debugging and batched operations).



```python
print(f"pred shape: {pred.shape}\ny shape: {y.shape}")
```

    pred shape: torch.Size([64, 1])
    y shape: torch.Size([64])


To compare the two matrix element-wise, we can use `eq()` function in PyTorch, however, the shape of two matrices should be the same. So, we reshape the targets `y` to be the same shape as `pred` which is (64, 1).




```python
y_aligned = y.view_as(pred)
print(f"y_aligned shape: {y_aligned.shape}")
```

    y_aligned shape: torch.Size([64, 1])



```python
pred_y_comparison = pred.eq(y_aligned)
print(f"Comparison result shape: {pred_y_comparison.shape}")
print(f"First 2 comparison results:\n{pred_y_comparison[0:2]}")
```

    Comparison result shape: torch.Size([64, 1])
    First 2 comparison results:
    tensor([[ True],
            [False]])


Now we compare the predicted classes with the actual classes (labels) for all examples in the batch. We compare them one by one, if they are equal, we count 1 and if not, we count 0. Then we sum them up the total predictions which then gives us the number of correct predictions for each batch.



```python
comparison_sum = pred_y_comparison.sum().item()
print(f"Sum of comparison results: {comparison_sum}")
```

    Sum of comparison results: 42


The above number is the total of all the ones and zeros in the matrix (1 for correct prediction and 0 for incorrect prediction). So, the above number is the total number of correct predictions for the entire batch.

To get the total number of correct predictions for the entire test dataset, we need to sum all the correct predictions for all batches together. See below for the complete code.

```py
correct += pred.eq(y.view_as(pred)).sum().item()
```


#### Putting Optimization and Evaluation Together
Now we can put everything together and create the training and evaluation loop. In here we define the number of epochs and then loop through the training and evaluation steps for each epoch.



```python
epochs = 3
for t in range(epochs):
    print(f"Epoch {t + 1}/{epochs}\n-------------------------------")
    train(model, device, train_dataloader, optimizer)
    test(model, device, test_dataloader)
    print()
```

    Epoch 1/3
    -------------------------------
    [Batch:    1/938 Processed:    64/60000 (0%)]	Loss: 2.088306
    [Batch:  100/938 Processed:  6400/60000 (11%)]	Loss: 2.057869
    [Batch:  200/938 Processed: 12800/60000 (21%)]	Loss: 2.107266
    [Batch:  300/938 Processed: 19200/60000 (32%)]	Loss: 1.994177
    [Batch:  400/938 Processed: 25600/60000 (43%)]	Loss: 2.041707
    [Batch:  500/938 Processed: 32000/60000 (53%)]	Loss: 2.025375
    [Batch:  600/938 Processed: 38400/60000 (64%)]	Loss: 1.961997
    [Batch:  700/938 Processed: 44800/60000 (75%)]	Loss: 2.017157
    [Batch:  800/938 Processed: 51200/60000 (85%)]	Loss: 1.940468
    [Batch:  900/938 Processed: 57600/60000 (96%)]	Loss: 1.910179
    Test Result:
     Average loss: 1.9061, Accuracy: 6872/10000 (69%)

    Epoch 2/3
    -------------------------------
    [Batch:    1/938 Processed:    64/60000 (0%)]	Loss: 1.903121
    [Batch:  100/938 Processed:  6400/60000 (11%)]	Loss: 1.846806
    [Batch:  200/938 Processed: 12800/60000 (21%)]	Loss: 1.933704
    [Batch:  300/938 Processed: 19200/60000 (32%)]	Loss: 1.755936
    [Batch:  400/938 Processed: 25600/60000 (43%)]	Loss: 1.818471
    [Batch:  500/938 Processed: 32000/60000 (53%)]	Loss: 1.790027
    [Batch:  600/938 Processed: 38400/60000 (64%)]	Loss: 1.701609
    [Batch:  700/938 Processed: 44800/60000 (75%)]	Loss: 1.795205
    [Batch:  800/938 Processed: 51200/60000 (85%)]	Loss: 1.674100
    [Batch:  900/938 Processed: 57600/60000 (96%)]	Loss: 1.635013
    Test Result:
     Average loss: 1.6227, Accuracy: 7177/10000 (72%)

    Epoch 3/3
    -------------------------------
    [Batch:    1/938 Processed:    64/60000 (0%)]	Loss: 1.630218
    [Batch:  100/938 Processed:  6400/60000 (11%)]	Loss: 1.540465
    [Batch:  200/938 Processed: 12800/60000 (21%)]	Loss: 1.665339
    [Batch:  300/938 Processed: 19200/60000 (32%)]	Loss: 1.444314
    [Batch:  400/938 Processed: 25600/60000 (43%)]	Loss: 1.499389
    [Batch:  500/938 Processed: 32000/60000 (53%)]	Loss: 1.467534
    [Batch:  600/938 Processed: 38400/60000 (64%)]	Loss: 1.378188
    [Batch:  700/938 Processed: 44800/60000 (75%)]	Loss: 1.510414
    [Batch:  800/938 Processed: 51200/60000 (85%)]	Loss: 1.365509
    [Batch:  900/938 Processed: 57600/60000 (96%)]	Loss: 1.320750
    Test Result:
     Average loss: 1.3028, Accuracy: 7525/10000 (75%)



### Inference
After we have trained our model and we are happy with the results, then we can make it ready for inference. The prerequisite for inference is that we need to be able to persist the model's parameters (weights and biases) to a storage device and then load them back later for inference.

PyTorch provides a simple way to save and load the model's parameters using `torch.save()` and `torch.load()` functions.



```python
# Save the model into a file
torch.save(model.state_dict(), "model.pth")
```

Loading the Model back from the file.


```python
loaded_model = NeuralNetwork().to(device)
loaded_model.load_state_dict(torch.load("model.pth", weights_only=True))
```




    <All keys matched successfully>




```python
loaded_model.eval()

# Inference for the first example in the test dataset
x = test_data[0][0]
y = test_data[0][1]

with torch.no_grad():
    x = x.to(device)
    logits = loaded_model(x)
    pred = logits.argmax(dim=1, keepdim=True)
    print(f"Predicted class: {pred.item()}, Actual class: {y}")
```

    Predicted class: 7, Actual class: 7
