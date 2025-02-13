# Multiclass Classification
Multiclass classification is a classification task that the target variable can take on more than two values. In other words, the target variable $y$ can take on $K$ different classes, where $K > 2$.

Some of the examples of multiclass classification problems include:

| Question | Target Variable | Classes |
| --- | --- | --- |
| What type of tumor is this? | Malignant, Benign, Normal | $K=3$ |
| Hand written digit recognition | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | $K=10$ |
| What type of animal is this? | Cat, Dog, Bird | $K=3$ |
| What type of vehicle is this? | Car, Truck, Bus, Motorcycle, Bicycle | $K=5$ |
| Next word (English) prediction in NLP | [Vocabulary of words/tokens](https://arxiv.org/abs/2406.16508) | $K=50,000+$ |


In binary classification, the target variable $y$ can take only two values, 0 or 1. That's why an algorithm like logistic regression which uses the sigmoid function is an appropriate choice for binary classification.

However, for multiclass classification, where $y$ can take on multiple values, we need a different algorithm such as **Softmax Regression** which is the generalization of logistic regression algorithm.

---
**In Binary Classification:**<br>

The out

$P(y=1|x;\theta) = h_{\theta}(x) = \frac{1}{1 + e^{-\theta^Tx}}$


**MultiClass Classification:**<br>
The output is a vector of $K$ values, where $K$ is the number of classes. The output vector is then passed through the softmax function to get the probabilities of each class.

Total probability of all classes should be 1.
$$P(y=1|x;\theta) + P(y=2|x;\theta) + P(y=3|x;\theta) + ... + P(y=K|x;\theta) = 1$$

Which can be written as:
$$\sum_{i=1}^{K} P(y=i|x;\theta) = 1$$

$P(y=i|x;\theta) = h_{\theta}(x) = \frac{e^{\theta_i^Tx}}{\sum_{j=1}^{K} e^{\theta_j^Tx}}$
