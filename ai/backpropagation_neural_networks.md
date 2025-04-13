---
date: "2025-02-16"
draft: false
title: "Backpropagation in Neural Networks"
description: "A deep dive into the backpropagation algorithm, computational graph, and how it works in neural networks."
tags:
    - "AI"
    - "Neural Networks"
---
Backpropagation is an algorithm to calculate the [derivative (gradient)](../math/derivatives.md) of the cost function with respect to the parameters of the neural network.

The name of the Backpropagation which is also called _back prop_ or _backward pass_, coming from the fact that after the [forward pass](neural_networks_inference.md) which calculates the output of the network based on the current parameters, then backprop calculates the derivative (gradient) of the cost function with respect to the parameters in the reverse order of the forward pass, meaning from the output layer back to the first layer. Hence, the name back propagation.


## Computational Graph
Computation graph is break down of the neural network (forward propagation and back propagation) into smaller building blocks, which then can be calculated step by step. This graph is a directed acyclic graph (DAG), and output of each node is calculated based on the output of the previous nodes. So, it can reuse the output of the previous nodes to calculate the output of the next nodes, which makes the computation more efficient.

See [Computational Graph](../ai/computational_graph_machine_learning.md) for more details.

In case of backpropagation, the computation graph is like breaking down the chain rule of the derivative into each composite function, and then calculate from the most outer function (loss function) to the most inner function (the parameters of the neural network).

Without the computational graph, we have to calculate the derivative of the loss function with respect to each parameter one by one which is very inefficient. However, with the computational graph, we start from the loss function (end of the graph) and calculate the derivative of the loss function with respect to the output of each function, then keep going backward to calculate the derivative of the loss function with respect to each parameter in the neural network. In this way, we reuse the output of the previous derivatives to calculate the next derivatives, which makes the computation more efficient.

> Note: In the backpropogation, we compute the gradient of the **loss** function with respect to each parameter, and then we average the gradients to find the **mean gradient** for that parameter over all samples. We'll use that mean gradient in the gradient descent algorithm to update the parameters at each iteration (step).

Let's go through this algorithm step by step using an example of a simple neural network like the one below:

![](images/nn_backpropagation.svg)

## Forward Propagation
We move forward (Left to Right) from the input node to the output node.

![](images/nn_computational_graph.svg)

## Loss Function
We continue to move forward (Left to Right) from the output node to the loss node.

## Backpropagation Algorithm
Now we move backwards (Right to Left), from the loss node to the input node.

![](images/nn_computational_graph.svg)


## Other Resources
- [Google ML Course - Backpropagation](https://developers.google.com/machine-learning/crash-course/backprop-scroll)
