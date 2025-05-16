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

## Establishing Baseline Level Performance
Establishing a baseline is a crucial step in the model evaluation process. A baseline model serves as a reference point to guide the development and improvement towards a better performance. If we don't have a baseline performance, we don't know the metrics that we are getting are good or bad and in what direction we need to improve.


In establishing a baseline model, can be established in multiple ways depending on the problem and our available data and resources. For example:
- **Human Baseline**: Use human performance as a baseline. This is often the best baseline to use, but it's not always available. For example, for image classification, or voice recognition, we can use human performance (and error rate) as a baseline. However, this is not always available.

- **External Models or Algorithms**: Use other existing models or algorithms as a baseline. This could be open-source models, or even competitor models that are available in the market.

- **Domain Knowledge**: Use domain knowledge of experts and SMEs to establish a baseline. This could be a good way if you have access to domain experts in the area you are training the model.

- **Simple Models**: If none of the above is available, we can ourselves train a simple model (based on the our guess, prior experience and domain knowledge) and use it as a baseline. This could be as simple as a linear regression model, a decision tree, or even a deep learning model.

## Diagnose and Address Bias and Variance Issues
See [Generalization](generalization_machine_learning.md) section for more details on bias and variance. High Bias (underfitting) and High Variance (overfitting) are two common problems in machine learning. Knowing how to identify and address these issues is crucial for building effective models.

See [Regularization](generalization_machine_learning.md#regularization) techniques and how to choose the right regularization parameters and create a balance between bias and variance. Also, see [Diagnosing Bias and Variance](generalization_machine_learning.md#diagnosing-bias-and-variance) for how to identify and address bias and variance issues during model evaluation.



## Performance Curves
Performance curves are series of plots that are used to visualize the relationship between model performance and different parameters and properties of the model. This can help in diagnosing bias and variance and understanding how different parameters affect the model performance.

The most common performance-parameter plots are:

- **Learning curves** – error (or accuracy) vs. training-set size (or training effort). As we saw above.
- **Training (loss) curves** – error (or accuracy) vs. number of epochs/iterations.
- **Validation curves** (aka hyperparameter curves) – error vs. hyperparameter value. e.g. [error-regularization curve](generalization_machine_learning.md#regularization)

> These plots are called **diagnostic plots** or **model evaluation curves**.
## Error Analysis
Error analysis is the process of analyzing the errors made by the model to understand the reasons or patterns behind the errors. This is usually done manually by examining the wrong predictions made by the model and trying to understand the reasons, patterns and underlying issues behind the errors. This step help us to know how to improve the model.

**Prioritizing Errors**<br>
Error analysis could show multiple overlapping issues and patterns. Be mindful of prioritizing the issues and patterns based on their impact and frequency. In some cases, solving one could takes a lot of time and effort, while that may have a small impact on the overall model performance.

## Model Tuning
Model tuning is the process of adjusting the hyperparameters, properties of the model, and the data setup to improve the model performance.

There are multiple ways to do this:

**Manual tuning**<br>
This approach is a common way to tune the hyperparameters. This is usually guided by the [Performance Curves](#performance-curves) plots which shows the relationship between the model performance (e.g. error or accuracy) and a hyperparameter value, or training set size, etc.

This approach is usually done by the data scientists and machine learning engineers who have experience with the model and the data. This is a time-consuming process and requires a lot of trial and error, and expertise.

**Automated tuning**<br>
This approach is a more systematic way to tune the hyperparameters. Instead of manually changing the hyperparameters, plotting the performance curves, and trying to find the best hyperparameters, we can use automated tools to do this for us. However, this approach is usually more expensive and time-consuming than the manual tuning, and it may not always yeild the best results.

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
