---
date: "2025-04-01"
draft: false
title: "Derivative of Matrices"
description: "A practical guide to matrix calculus using Jacobians and partial derivatives for both scalar and vector-valued functions."
tags:
    - "Math"
    - "Linear Algebra"
    - "Calculus"
---

When we take derivative of single variable in calculus, we are finding the effect of a small change in the variable $x$ on the function $f(x)$.

$$\frac{df}{dx} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

We can extend this idea to matrices. A matrix is a collection of variables (multivariate). So, we can think of the derivative of a matrix as a collection of derivatives of each variable in the matrix.

For example for the function $F(A)$ where $A$ is a matrix, we can write the derivative as:

$$\frac{dF}{dA} = \begin{bmatrix}
\frac{\partial F}{\partial a_{11}} & \frac{\partial F}{\partial a_{12}} & \cdots & \frac{\partial F}{\partial a_{1n}} \\
\frac{\partial F}{\partial a_{21}} & \frac{\partial F}{\partial a_{22}} & \cdots & \frac{\partial F}{\partial a_{2n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial F}{\partial a_{m1}} & \frac{\partial F}{\partial a_{m2}} & \cdots & \frac{\partial F}{\partial a_{mn}}
\end{bmatrix}$$
Where
- $A$ is an $m \times n$ matrix
- $a_{ij}$ is the $(i,j)$-th element of the matrix $A$.
- $F(A)$ is a function of the matrix $A$.

This is called the **Jacobian matrix**. The Jacobian matrix is a matrix of all first-order partial derivatives of a vector-valued function. It describes how the function changes as the input changes.

**Scalar-valued function of a matrix**:<br>

- Input: a matrix $A \in \mathbb{R}^{m \times n}$
- Output: a scalar $F(A) \in \mathbb{R}$

In this case, the derivative $\frac{dF}{dA}$ is a matrix of **partial derivatives**. It has the **same shape as $A$**:

$$
\left[ \frac{\partial F}{\partial a_{ij}} \right]_{m \times n}
$$


**Vector-valued function of a matrix**:

- Input: a matrix $A \in \mathbb{R}^{m \times n}$
- Output: a matrix $F(A) \in \mathbb{R}^{p \times q}$

Then the derivative $\frac{dF}{dA}$ is a **4D tensor** of shape $(p \times q) \times (m \times n)$:

$$
\frac{\partial F}{\partial A} = \left[ \frac{\partial f_{kl}}{\partial a_{ij}} \right] \in \mathbb{R}^{m \times n \times p \times q}
$$

In practice, to simplify the computation of this 4D tensor, we can **flatten** the input and output matrices into vectors. Flattening (also called **vectorization**) can be done in two ways:

**Row-major order**:<br>
Stack the rows of the matrix into a single column vector.

For example, matrix $A$:
$$
A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

$A$ with $(3 \times 3)$ shape can be flattened into a vector $a$ of shape $(9 \times 1)$:

$$
\text{vec}(A) = \begin{bmatrix}
1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9
\end{bmatrix}
$$

**Column-major order**:<br>
Stack the columns of the matrix into a single column vector.

For example, the same matrix $A$ can be flattened into a vector $a$:

$$
\text{vec}(A) = \begin{bmatrix}
1 & 4 & 7 & 2 & 5 & 8 & 3 & 6 & 9
\end{bmatrix}
$$

Since we are going to use Python and PyTorch in practice, we follow thier convention of **row-major order**. Remember, that whatever convention you choose, you need to be consistent throughout your entire calculations.


Each row of Jacobian matrix is called **a gradient vector** of the function with respect to the input matrix.


## Scalar-valued Function of a Matrix
This is the case when the function $F(A)$ is a scalar value (not a matrix). In this case, the Jacobian matrix is a row vector.

For example, let's say we have the following that takes a matrix $A$ and returns the sum of all the elements in the matrix multiplied by $2$. So, the output is a scalar value.

$$F(A) = 2\,\text{sum}(A)$$

Which can be written as:

$$F(A) = 2\sum _{i=1}^{m}\sum_{j=1}^{n} a_{ij}$$



If we have matrix $A$ in shape of $2 \times 3$:
$$A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23}
\end{bmatrix}$$

We need to broadcast $2$ to the shape of $A$ to perform the multiplication. So we can think of $F(A)$ as:
$$F(A) = \begin{bmatrix} 2 & 2 & 2 \\ 2 & 2 & 2 \end{bmatrix} \odot
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix}$$

Then $F(A)$ is a scalar value:

$$F(A) = 2a_{11} + 2a_{12} + 2a_{13} + 2a_{21} + 2a_{22} + 2a_{23}$$

Now, let's calculate the derivative of $F(A)$ with respect to $A$:

$$\frac{dF}{dA} = \begin{bmatrix}
\frac{\partial F}{\partial a_{11}} & \frac{\partial F}{\partial a_{12}} & \frac{\partial F}{\partial a_{13}} \\
\frac{\partial F}{\partial a_{21}} & \frac{\partial F}{\partial a_{22}} & \frac{\partial F}{\partial a_{23}}
\end{bmatrix}$$

We can simply take the derivative of $F(A)$ with respect to each element of the matrix $A$:

For the first row:
$$\frac{\partial F}{\partial a_{11}} = 2 \quad \frac{\partial F}{\partial a_{12}} = 2 \quad \frac{\partial F}{\partial a_{13}} = 2$$

For the second row:
$$\frac{\partial F}{\partial a_{21}} = 2 \quad \frac{\partial F}{\partial a_{22}} = 2 \quad \frac{\partial F}{\partial a_{23}} = 2$$

So, the Jacobian matrix is:

$$\frac{dF}{dA} = \begin{bmatrix} 2 & 2 & 2 \\ 2 & 2 & 2 \end{bmatrix}$$

**Example:**

Let's say we have the following matrix $A$:

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$$

The $F(A)$ is a scalar-valued function of the matrix $A$:

$$F(A) = 2\times 1 + 2\times 2 + 2\times 3 + 2\times 4 + 2\times 5 + 2\times 6 = 42$$

Let's use PyTorch `autograd` for our derivative calculation.



```python
import torch

A = torch.tensor(
    [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True
)

F = (2 * A).sum()

print(f"A:\n{A}")
print(f"Z:\n{F}")
```

    A:
    tensor([[1., 2., 3.],
            [4., 5., 6.]], requires_grad=True)
    Z:
    42.0


Now we use PyTorch computional graph to go backward to the graph and calculate the derivative of $F$ with respect to $A$.



```python
F.backward()
```


```python
print(f"\nA_t.grad:\n{A.grad}")
```


    A_t.grad:
    tensor([[2., 2., 2.],
            [2., 2., 2.]])


## Vector-valued Function of a Matrix

Let's say we have the following function:

$$Z(A,W,B) = A.W + B$$

And we want to calculate the partial derivative of $Z$ with respect to $W$.

$$\frac{\partial Z}{\partial W} = ?$$

Function $Z$ takes a matrix $A$, $W$ and $B$ as inputs. The output is a matrix $Z$.

Let's say we have the following matrices:

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad
W = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \end{bmatrix}, \quad
B = \begin{bmatrix} 13 & 14 \end{bmatrix}$$



$Z$ is calculated as:

$$Z = \begin{bmatrix}
1 & 2 & 3 \\ 4 & 5 & 6
\end{bmatrix}
\cdot
\begin{bmatrix}
w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32}
\end{bmatrix}
+ \begin{bmatrix}
13 & 14
\end{bmatrix}$$


Which is a matrix of shape $2 \times 2$:

$$Z = \begin{bmatrix}
1w_{11} + 2w_{21} + 3w_{31} + 13 & 1w_{12} + 2w_{22} + 3w_{32} + 14 \\
4w_{11} + 5w_{21} + 6w_{31} + 13 & 4w_{12} + 5w_{22} + 6w_{32} + 14
\end{bmatrix}$$


Derivative of $Z$ with respect to $W$ is:

$$\frac{\partial Z}{\partial W} = \begin{bmatrix}
\frac{\partial Z}{\partial w_{11}} & \frac{\partial Z}{\partial w_{12}} \\
\frac{\partial Z}{\partial w_{21}} & \frac{\partial Z}{\partial w_{22}} \\
\frac{\partial Z}{\partial w_{31}} & \frac{\partial Z}{\partial w_{32}}
\end{bmatrix}$$

As we discussed earlier, $Z$ by itself is a matrix of shape $2 \times 2$, and our goal is to find how every element $z_{kl}$ in $Z$ changes with respect to an element of $w_{ij}$ in $W$. However, this matrix is 4D tensor of shape $(2 \times 2) \times (3 \times 2)$.

We have the following:
- $Z \in \mathbb{R}^{2 \times 2}$
- $W \in \mathbb{R}^{3 \times 2}$
- We want to compute $\frac{dZ}{dW}$

Since each element $z_{kl}$ in $Z$ can depend on every element $w_{ij}$ in $W$, the **full derivative** really is:

$$
\frac{\partial Z}{\partial W} = \left[ \frac{\partial z_{kl}}{\partial w_{ij}} \right] \in \mathbb{R}^{2 \times 2 \times 3 \times 2}
$$

So, we have to flatten (vectorize) the input and output matrices into vectors and create a Jacobian matrix of shape $((2 \times 2) \times (3 \times 2)) = (4 \times 6)$.

**Vectorize (Flatten) the output $Z$**<br>
Turn the matrix $Z \in \mathbb{R}^{m \times n}$ into a vector of size **$mn$**:

$$
\text{vec}(Z) = \begin{bmatrix} z_{11} & z_{12} & z_{21} & z_{22} \end{bmatrix}
$$

> Note: In this example, we use row-major order to flatten the matrix to a vector.

**Vectorize (Flatten) the input $W$**<br>
Similarly, flatten the matrix $W \in \mathbb{R}^{p \times q}$ into a vector of size **$pq$**:

$$
\text{vec}(W) = \begin{bmatrix} w_{11} & w_{12} & w_{21} & w_{22} & w_{31} & w_{32} \end{bmatrix}
$$


**Build the Jacobian Matrix**:<br>

The 2D Jacobian has shape $(mn) \times (pq)$, which is $(4 \times 6)$ in this case.

$$
\frac{\partial\, \text{vec}(Z)}{\partial \,\text{vec}(W)} \in \mathbb{R}^{4 \times 6}
$$

We can write the Jacobian as:

$$
\frac{\partial\,\text{vec}(Z)}{\partial\,\text{vec}(W)} =
\begin{bmatrix}
\frac{\partial z_{11}}{\partial w_{11}} & \frac{\partial z_{11}}{\partial w_{12}} & \frac{\partial z_{11}}{\partial w_{21}} & \frac{\partial z_{11}}{\partial w_{22}} & \frac{\partial z_{11}}{\partial w_{31}} & \frac{\partial z_{11}}{\partial w_{32}} \\
\frac{\partial z_{12}}{\partial w_{11}} & \frac{\partial z_{12}}{\partial w_{12}} & \frac{\partial z_{12}}{\partial w_{21}} & \frac{\partial z_{12}}{\partial w_{22}} & \frac{\partial z_{12}}{\partial w_{31}} & \frac{\partial z_{12}}{\partial w_{32}} \\
\frac{\partial z_{21}}{\partial w_{11}} & \frac{\partial z_{21}}{\partial w_{12}} & \frac{\partial z_{21}}{\partial w_{21}} & \frac{\partial z_{21}}{\partial w_{22}} & \frac{\partial z_{21}}{\partial w_{31}} & \frac{\partial z_{21}}{\partial w_{32}} \\
\frac{\partial z_{22}}{\partial w_{11}} & \frac{\partial z_{22}}{\partial w_{12}} & \frac{\partial z_{22}}{\partial w_{21}} & \frac{\partial z_{22}}{\partial w_{22}} & \frac{\partial z_{22}}{\partial w_{31}} & \frac{\partial z_{22}}{\partial w_{32}}
\end{bmatrix}
$$


**Rows** = one element of $\text{vec}(Z)$ with respect to all elements of $\text{vec}(W)$
**Columns** = all elements of $\text{vec}(Z)$ with respect to one element of $\text{vec}(W)$


We can see it as just a table of **all the partial derivatives** of each output element with respect to each input element.


Recall this is $Z$:

$$Z = \begin{bmatrix}
1w_{11} + 2w_{21} + 3w_{31} + 13 & 1w_{12} + 2w_{22} + 3w_{32} + 14 \\
4w_{11} + 5w_{21} + 6w_{31} + 13 & 4w_{12} + 5w_{22} + 6w_{32} + 14
\end{bmatrix}$$

So, the partial derivatives for the first row of Jacobian matrix are:

$$
\frac{\partial z_{11}}{\partial w_{11}} = 1, \quad
\frac{\partial z_{11}}{\partial w_{12}} = 0, \quad
\frac{\partial z_{11}}{\partial w_{21}} = 2, \quad
\frac{\partial z_{11}}{\partial w_{22}} = 0, \quad
\frac{\partial z_{11}}{\partial w_{31}} = 3, \quad
\frac{\partial z_{11}}{\partial w_{32}} = 0
$$

If we continue this for all other rows of the Jacobian, we get the matrix of partial derivatives of $Z$ with respect to $W$:

$$
\frac{\partial\, \text{vec}(Z)}{\partial\,\text{vec}(W)} =
\begin{bmatrix} 1 & 0 & 2 & 0 & 3 & 0 \\
0 & 1 & 0 & 2 & 0 & 3 \\
4 & 0 & 5 & 0 & 6 & 0 \\
0 & 4 & 0 & 5 & 0 & 6
\end{bmatrix}
$$

## Chain Rule

Let's say we have $Z$ which is a function of matrix $A$, $W$ and $B$, similar to the previous example:

$$Z = A.W + B$$

And we also have function $S$ which is a function of $Z$:
$$S = \sum_{i=1}^{m}\sum_{j=1}^{n} z_{ij}$$

Now we want to calculate the derivative of $S$ with respect to $W$ using the chain rule:

$$\frac{\partial S}{\partial W} = \frac{\partial S}{\partial Z} \cdot \frac{\partial Z}{\partial W}$$

We use the same matrix $A$, $W$ and $B$ as before:

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad
W = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \end{bmatrix}, \quad
B = \begin{bmatrix} 13 & 14 \end{bmatrix}$$

Recall that we calculated $Z$ as:

$$Z = \begin{bmatrix}
1w_{11} + 2w_{21} + 3w_{31} + 13 & 1w_{12} + 2w_{22} + 3w_{32} + 14 \\
4w_{11} + 5w_{21} + 6w_{31} + 13 & 4w_{12} + 5w_{22} + 6w_{32} + 14
\end{bmatrix}$$

Now the function $S$ which is scalar valued function of $Z$ is:

$$S = \sum_{i=1}^{m}\sum_{j=1}^{n} z_{ij} = z_{11} + z_{12} + z_{21} + z_{22}$$

Which we can calculate as:

$$S = z_{11} + z_{12} + z_{21} + z_{22}$$

And if we substitute the values of $z_{ij}$ we have:

$$S = (1 + 4)w_{11} + (2 + 5)w_{21} + (3 + 6)w_{31} + (1 + 4)w_{12} + (2 + 5)w_{22} + (3 + 6)w_{32} + 54$$


Derivative of $S$ with respect to $Z$ is:

$$\frac{\partial S}{\partial Z} = \begin{bmatrix}
\frac{\partial S}{\partial z_{11}} & \frac{\partial S}{\partial z_{12}} \\
\frac{\partial S}{\partial z_{21}} & \frac{\partial S}{\partial z_{22}}
\end{bmatrix} = \begin{bmatrix}
1 & 1 \\ 1 & 1
\end{bmatrix}$$


We have already calculated the derivative of $Z$ with respect to $W$ in the previous example:

$$
\frac{\partial \text{vec}(Z)}{\partial\text{vec}(W)} =
\begin{bmatrix} 1 & 0 & 2 & 0 & 3 & 0 \\
0 & 1 & 0 & 2 & 0 & 3 \\
4 & 0 & 5 & 0 & 6 & 0 \\
0 & 4 & 0 & 5 & 0 & 6
\end{bmatrix}
$$



Recall that the derivative of $S$ with respect to $W$ is:

$$
\frac{\partial S}{\partial W} = \frac{\partial S}{\partial Z} \cdot \frac{\partial\,\text{vec}(Z)}{\partial\, \text{vec}(W)}
$$

We already calculated $\frac{\partial S}{\partial Z}$:
$$
\frac{\partial S}{\partial Z} = \begin{bmatrix}
1 & 1 \\ 1 & 1
\end{bmatrix}
$$

We used the **row-major order** to vectorize the matrices. So, we'll use the same order to vectorize the $\frac{\partial S}{\partial Z}$ matrix.

$$
\frac{\partial S}{\partial Z} = \begin{bmatrix}
1 & 1 & 1 & 1
\end{bmatrix}
$$

So, now we can calculate the derivative of $S$ with respect to $W$:

$$
\frac{\partial S}{\partial W} = \begin{bmatrix}
1 & 1 & 1 & 1
\end{bmatrix}
\cdot
\begin{bmatrix} 1 & 0 & 2 & 0 & 3 & 0 \\
0 & 1 & 0 & 2 & 0 & 3 \\ 4 & 0 & 5 & 0 & 6 & 0 \\ 0 & 4 & 0 & 5 & 0 & 6
\end{bmatrix}
= \begin{bmatrix}
5 & 5 & 7 & 7 & 9 & 9
\end{bmatrix}
$$

> Note that the operation here is the dot product of the two matrices (matrix multiplication, not element-wise multiplication). So, the result is a row vector of size $1 \times 6$.

Now we have to reshape the output to the shape of $W$ which is $3 \times 2$ (reverse of flatteing) using the same order:

$$
\frac{\partial S}{\partial W} = \begin{bmatrix}
5 & 5 \\ 7 & 7 \\ 9 & 9 \end{bmatrix}
$$


Let's use PyTorch `autograd` to calculate the derivative of $S$ with respect to $W$ and compare it with our result.


```python
W = torch.tensor(
    [[7, 8], [10, 11], [12, 13]], dtype=torch.float32, requires_grad=True
)
B = torch.tensor([13, 14], dtype=torch.float32, requires_grad=True)

Z = torch.matmul(A, W) + B
S = Z.sum()

print(f"Z_t:\n{Z}")
print(f"S:\n{S}")
```

    Z_t:
    tensor([[ 76.,  83.],
            [163., 179.]], grad_fn=<AddBackward0>)
    S:
    501.0



```python
S.backward()
```


```python
print(f"\nW.grad:\n{W.grad}")
```


    W.grad:
    tensor([[5., 5.],
            [7., 7.],
            [9., 9.]])
