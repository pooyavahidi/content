---
date: "2024-10-23"
draft: false
title: "Model Evaluation in Machine Learning"
description: "An overview of model evaluation techniques in machine learning, including cross-validation and performance metrics."
tags:
    - "AI"
---
Model evaluation involves the methods and metrics (e.g., cross-validation, test accuracy, precision-recall) used to assess how well a machine learning model performs on unseen (real world) data. This is called [generalization](generalization_machine_learning.md).

Model evaluation provides the tools to measure generalization. For instance, evaluating performance on a validation or test set helps estimate how well the model generalizes. It serves as the practical mechanism to verify generalization.

Evaluation process could be seen as a time-consuming process, but it more often than not saves a lot of time and effort in the long run.

**Training a Model is an Iterative Process**<br>
Training a model is an iterative process. It's very unlikely that the trained model at first attempt deliver the desired performance. The trained model needs to be evaluated, and based on the evaluation results, the model needs to be improved. The cycle of **training>evaluation>improvement** will be repeated until the model performance is satisfactory, and even when it's moved to production, the model still needs to be monitored and improved and repeat the cycle. The model training is a continuous process, even after the model is deployed in production.


![](images/model_evaluation_process.svg)

After each time evaluation (or even after when model is deployed in production), we need to evaluate the model performance and based on the evaluation results, we may need to improve the model. The improvement could be:
- Tune the hyperparameters on the current model
- Change the model architecture or algorithm
- Change the data setup (it could be only applying feature engineering or even going back further to the data engineering step to collect more data or to preprocess the data differently).
- Train different models with the same data setup or different data setup and compare the results. It's a common to train and evaluate multiple models that include different data setup and algorithms and compare the results. This is also called **model selection**.
- Change the target (i.e. the prediction).
- etc


**Model Evaluation Steps**<br>

After the data is prerpocessed and we are ready to train a model, then we:
1. Splitting the Data for Cross-Validation
2. Choosing the Right Evaluation Metrics
3. Baseline the Trained Model
4. Cycle of Training-Evaluation-Improvement

**Model Evaluation Goal**<br>
After initial steps, we keep iterating through steps of model improvement and tuning with the goal of improving the model performance. In other words, increasing the model's ability to [generalizes](generalization_machine_learning.md) better.

> Model performance is equal to the model's ability to generalize to unseen data. So, we use these terms interchangeably

During evaluation, we usually are fighting against [underfitting](generalization_machine_learning.md#underfitting) (high bias) and [overfitting](generalization_machine_learning.md#overfitting) (high variance). The main goal of model evaluation is to find the right balance between underfitting and overfitting which is called **bias-variance tradeoff**.

Therefore we keep this goal in mind and constantly let the model's bias and variance guide us in the process of model evaluation.

**Other Aspects to Consider**<br>
There other important aspects of improving a machine learning model which may not be the exact goals of model evaluation, but they are important specially in practical applications. These include:
- Model Interpretability: Understanding how the model makes predictions and ensuring it aligns with domain knowledge.
- Efficiency: The efficiency of the model in both training (and re-training) and inference both in terms of time and compute resources.
- Scalability
- etc

## Splitting the Data for Cross-Validation
The first step in model evaluation is to split the data for [cross-validation](cross_validation_machine_learning.md). Cross-validation helps us to evaluate our model after each improvement to see how well it [generalizes](generalization_machine_learning.md) ( how well it performs on unseen data).

## Choosing the Right Evaluation Metrics
Evaluation metrics are used to measure the performance of a model on validation sets. The choice of metrics depends on the type of problem (regression, classification, clustering, etc.) and the specific problem we are trying to solve. See [Evaluation Metrics](evaluation_metrics_machine_learning.md) for more details.

## Bias and Variance
See [Generalization](generalization_machine_learning.md) section for more details on bias and variance. High Bias (underfitting) and High Variance (overfitting) are two common problems in machine learning. Knowing how to identify and address these issues is crucial for building effective models.

Also, see [Regularization](generalization_machine_learning.md#regularization) techniques and how to choose the right regularization parameters and create a balance between bias and variance.

## Baseline Model
Establishing a baseline is a crucial step in the model evaluation process. A baseline model serves as a reference point to guide the development and improvement towards a better performance. If we don't have a baseline performance, we don't know the metrics that we are getting are good or bad and in what direction we need to improve.


In establishing a baseline model, can be established in multiple ways depending on the problem and our available data and resources. For example:
- **Human Baseline**: Use human performance as a baseline. This is often the best baseline to use, but it's not always available. For example, for image classification, or voice recognition, we can use human performance (and error rate) as a baseline. However, this is not always available.

- **External Models**: Use other existing models or algorithms as a baseline. This could be open-source models, or even competitor models that are available in the market.

- **Domain Knowledge**: Use domain knowledge of experts and SMEs to establish a baseline. This could be a good way if you have access to domain experts in the area you are training the model.

- **Simple Models**: If none of the above is available, we can ourselves train a simple model (based on the our guess, prior experience and domain knowledge) and use it as a baseline. This could be as simple as a linear regression model, a decision tree, or even a deep learning model.


## Diagnosing Bias and Variance

After we establish a baseline model, trained our new model, and evaluated the model performance on the validation set, we need to check if the model is underfitting (high bias) or overfitting (high variance). This is done by comparing the performances of baseline, model training, and model evaluation.

Diagnosing bias and variance is an ongoing process, meaning that we diagnose the bias and variance after each cycle of training-evaluation-improvement to see where are we in the trade off of bias and variance.

Let's say we have a regression model and we use Mean Squared Error (MSE) as the evaluation metric.
We have 3 performance metrics:
- **Baseline Performance Error**: Percentage of errors in the baseline model.
- **Training Error**: Percentage of model errors on the training set.
- **Validation Error**: Percentage of model errors on the validation set.

> Baseline performance error as discussed could be a human performance, external model performance, or a simple model performance. So, depending on where did it come from it could be a starting point or the ideal model performance.
>
> For example, if the baseline performance is the result of a expert human performance, then the baseline performance is the **ideal** model performance and could be used as our **goal**. However, if the baseline performance is the result of a simple model which we trained initially, then the baseline performance is just a **starting point**.

Let's assume in this example our baseline performance is the **ideal** model performance.

**Underfitting (High Bias)**:<br>
This happens when the gap between the baseline and training error is high. This means that the model is not capable (too simple) of learning the training data.

$$\text{Baseline Error} \ll \text{Training Error}$$

**Overfitting (High Variance)**:<br>
This happens when the gap between the **training error** and **validation error** is high. This means that the model is too complex and is learning the noise in the training data, which performs well on the training set but poorly on the validation set (unseen data).

$$\text{Training Error} \ll \text{Validation Error}$$

**Example**:<br>
Let's say we have the following scenarios:

| Bias-Variance | Baseline Error | Training Error | Validation Error |
|-|-|-|-|
| High Variance (Overfitting) | 10% | 11% | 20%
| High Bias (Underfitting) | 10% | 20% | 25%
| Good Fit | 10% | 11% | 12%

> Note: The above numbers are just for illustration purposes. The actual thresholds for high bias, high variance, and good fit will depend on the problem and the data and is different case by case. For example, in some cases, even 1% difference in the training and validation error could be considered as overfitting, etc.

In case of underfitting, we usually don't need to even progress to the validation set. We can conclude the high bias from comparison of the baseline and training error without even evaluating the model on the validation set.

### Learning Curves
Learning curves are a powerful tool for diagnosing bias and variance in machine learning models. They plot the training and validation error as a function of the training set size. By analyzing this curve, we can gain insights into the model's performance and identify potential issues.

See [Increasing Training Set Size and Learning Curves](generalization_machine_learning.md#increase-training-data) for more details.


### Bias and Variance in Linear Regression

Let's say we have a regularized linear regression model as follow:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2$$

Where:
- $J(\theta)$ is the cost function
- $h_\theta(x^{(i)})$ is the model prediction (also noted as $y^{(i)}$ or $f_\theta(x^{(i)})$)
- $y^{(i)}$ is the actual value
- $m$ is the number of training examples
- $n$ is the number of features
- $\lambda$ is the regularization parameter
- $\theta$ is the model parameters (weights)
- $x^{(i)}$ is the input feature vector for the $i^{th}$ training example

Let's say this model has unacceptable large errors in prediction, what are the possible actions to fix this? First we need to diagnose the problem if it's high bias (underfitting) or high variance (overfitting). We can do this by comparing the training and validation errors and learning curves explained above. Then to solve the problem, we can take the following actions:

|Action|Result|
|-|-|
|Getting more training data| Fixes high variance|
|Reducing the features| Fixes high variance|
|Getting more features| Fixes high bias|
|Increasing model's complexity (e.g. adding polynomial features)| Fixes high bias|
|Increasing $\lambda$| Fixes high Variance|
|Decreasing $\lambda$| Fixes high bias|

There are other actions which could be taken to fix the high bias and high variance, but the point of above example, is to show that more often than not, we are dealing with high bias (overfitting) and high variance (underfitting) and our task is to first diagnose the problem and then take the right action to fix it (i.e. balance the bias and variance).

### Bias and Variance in Neural Networks
[Neural Networks](neural_networks_overview.md) by design are much more capable of learning complex patterns in the data. They are very flexible in terms of the model architecture which allows us to create as complex models as we want. So, because of this ability, in the tradeoff between bias and variance, neural networks are usually more prone to overfitting (high variance) than underfitting (high bias).

This is why large neural networks are **low bias** machines. If we make a neural network large enough, we can almost always fit (balance between bias and variance) the training data well. This is one of the reasons behind the increasing popularity of deep learning.

We can use the following guidance to diagnose bias and variance in neural networks and take the right actions to fix the problem


![](images/nn_training_evaluation_flow_bias_variance.svg)

In the above flow:
- Larger Network: It means either adding more hidden layers (deeper) or adding more neurons in the hidden layers (wider), or both. Also other factors such as types of the layers, activation functions, number of epochs, batch size, and other hyperparameters can be explored to increase the model capability.
- Both increasing the size of the neural network and increasing the training data have limits. As networks get larger, they are much more demanding in terms of computational resources and time. Also, adding more data may not always be possible.

**Small or Large Neural Networks**<br>
Intuitively, we may think that the larger neural networks (being more complex) are prone to overfitting (high variance). It turns out that a large neural network usually do as well or better than a small neural network as long as the [regularization](generalization_machine_learning.md#regularization-in-neural-networks) is used properly.

Therefore, it's a good idea to go for a larger networks as much as your computational resources, budget and time allow.


## Performance Curves
Performance curves are series of plots that are used to visualize the relationship between model performance and different parameters and properties of the model. This can help in diagnosing bias and variance and understanding how different parameters affect the model performance.

The most common performance-parameter plots are:

- **Learning curves** – error (or accuracy) vs. training-set size (or training effort). As we saw above.
- **Training (loss) curves** – error (or accuracy) vs. number of epochs/iterations.
- **Validation curves** (aka hyperparameter curves) – error vs. hyperparameter value. e.g. [error-regularization curve](generalization_machine_learning.md#regularization)

> These plots are called **diagnostic plots** or **model evaluation curves**.

## Model Tuning
After evaluating the model we may need to tune our model by changing the hyperparameters. There are multiple ways to do this:

- Manual tuning (need extensive domain knowledge and experience with the model).

- Grid search (try all possible combinations of hyperparameters). This approach trains the model in all possible combinations of the hyperparameters and compare the results. This is computationally expensive and time-consuming.

- Random search (try random combinations of hyperparameters). This approach is a subset of the grid search, and it's more efficient and faster than the grid search. It randomly selects the hyperparameters and trains the model.

- Automated hyperparameter tuning (use tools like AWS SageMaker, or other tools like Optuna, Ray Tune, Keras Tuner, etc). This approach uses optimization algorithms to find the best hyperparameters.

- Bayesian optimization (use tools like Optuna, Hyperopt, etc). This approach uses Bayesian optimization to find the best hyperparameters.

## Model Selection
Model selection is the process of training multiple models using the same or different data setups and comparing the results. This can help in improving the current model performance or choosing the best model for the task. There are multiple ways to compare the models and select the best one:

- Compare the performance of the models using the evaluation metrics. This is the most common way to compare the models.
- Learning Curves: Plot learning curves to compare models and check for overfitting or underfitting.
- Validation Curves: Use validation curves to understand how model performance changes with different hyperparameter values.

> Note: As discussed above, we use **validation set**, not the **test set**, to compare the models. The test set is used only at the end of the model selection process to estimate the performance of the best model on unseen data.

## Keep Track of Experiments
Use tools like MLflow, TensorBoard, or Weights & Biases to log experiments, track metrics, and compare models systematically.

## Continuous Evaluation
Machine learning models can degrade over time as data evolves. Implement a strategy for continuous monitoring and evaluation to ensure models remain effective.

## Ensemble Methods
Sometimes, instead of making a perfect single model, it's better to combine multiple models to get better results. Leveraging ensemble methods like bagging, boosting, and stacking can improve model performance by combining multiple models' predictions.

## Automated Tools
Consider using automated machine learning (AutoML) tools that can automatically preprocess data, select models, and tune hyperparameters. Examples include Auto-sklearn, TPOT, and H2O.
