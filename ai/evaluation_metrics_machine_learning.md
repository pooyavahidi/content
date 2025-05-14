---
date: "2025-03-04"
draft: false
title: "Evaluation Metrics in Machine Learning"
description: "An overview of evaluation metrics used in machine learning to assess model performance."
tags:
    - "AI"
---
Evaluation metrics are used to assess the performance of a trained model on _valiation sets_ (also called _cross validation sets_). See [cross-validation](cross_validation_machine_learning.md) for more details.

So, at this stage, we have trained one or multiple models using the training set, and now we want to evaluate the model performance using the validation set. We choose the right evaluation metrics based on the type of the model and the task.


## Regression Metrics
The common metrics for regression tasks include:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing an error metric in the same units as the target variable.
- **R-squared (R^2)**: Represents the proportion of variance in the target variable that can be explained by the model. A higher R^2 indicates a better fit.

**Mean Squared Error (MSE)**<br>
This metric is exactly what we used in the cost function of the linear regression model during the training process. So, it means we evaluate the model performance using the same metric, but on the _cross validation set_.

$$J_{cv}(\theta) = \frac{1}{m} \sum_{i=1}^{m} (f_\theta({x_{cv}}^{(i)}) - {y_{cv}}^{(i)})^2$$

Where:
- $J_{cv}(\theta)$ is the cost function on the cross-validation set on the model parameters $\theta$.
- $m$ is the number of samples in the cross-validation set.
- $f_\theta({x_{cv}}^{(i)})$ is the predicted value of the model on the cross-validation set.
- ${y_{cv}}^{(i)}$ is the actual value of the model on the cross-validation set.

> Note: In evaluating the model performance using error metrics, we don't use the regularization term. The regularization term is used to prevent overfitting during the training process, but when we evaluate the model performance, we only use the cost function (e.g. MSE) without the regularization term.


## Classification Metrics
Accuracy, Precision, Recall, F1 Score, ROC-AUC

> **Cost-Sensitive Evaluation:** In some applications, the cost of false positives and false negatives can be different. Adjust your evaluation metrics to reflect these costs.

## Clustering Metrics
Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index

## Time Series Metrics
Mean Absolute Error (MAEP), Mean Squared Error (MSE). Root Mean Squared Error (RMSE),
Mean Absolute Percentage Error (MAPE), Symmetric Mean Absolute Percentage Error (SMAPE)

**Domain-Specific Considerations:** Some domains may require specific evaluation strategies or metrics.

## Choosing the Right Metric
- Metrics like MAE, RMSE, and MAPE provide a standardized way of measuring error across different models. They quantify how far off predictions are from actual values, regardless of the underlying model. These metrics allow for an objective comparison of model performance by providing a single, quantifiable measure of error. This facilitates the evaluation of which model performs better on a specific task.

- In scenarios with highly imbalanced data, precision, recall, and F1 score might be more informative for classification tasks because they provide a clearer picture of how well the model performs on the minority class

- Probabilistic Models: For models that output probabilities (e.g., some classifiers), metrics like log loss or AUC-ROC might provide additional insights into model performance beyond what MAE or RMSE can offer.
