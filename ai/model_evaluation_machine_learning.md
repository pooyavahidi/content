---
date: "2024-10-23"
draft: true
title: "Model Evaluation in Machine Learning"
description: "An overview of model evaluation techniques in machine learning, including cross-validation and performance metrics."
tags:
    - "AI"
---
Model evaluation involves the methods and metrics (e.g., cross-validation, test accuracy, precision-recall) used to assess the [generalization](generalization_machine_learning.md) performance and ensure that the model is not overfitting or underfitting.

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


**Model Evaluation Initial Steps**<br>

After the data is prerpocessed and we are ready to train a model, then we:
1. Splitting the Data (for training and testing)
2. Choosing the Right Evaluation Metrics
3. Baseline the Trained Model

**Model Evaluation Goal**<br>
After initial steps, we keep iterating through steps of model improvement and tuning with the goal of improving the model performance. The model performance simply means how well the model [generalizes](generalization_machine_learning.md) to unseen data.

So, in most cases, we are fighting against [underfitting](generalization_machine_learning.md#underfitting) (high bias) and [overfitting](generalization_machine_learning.md#overfitting) (high variance). The main goal of model evaluation is to find the right balance between underfitting and overfitting. This is also called the **bias-variance tradeoff**.

So, in all the steps of model evaluation, have this goal in mind and constantly let the model's bias and variance guide you in the process of model evaluation.

## Splitting the Data
Splitting the data is a crucial initial step in machine learning to ensure that the model can generalize well to unseen data. The data is usually divided into two sets (two-way split) or three sets (three-way split):


**Two-way Splits (Training/Test):**<br>
When using only training and test sets:

- 80/20 split: 80% training, 20% testing (common approach)
- 70/30 split: 70% training, 30% testing (used when more evaluation data is needed)
- 90/10 split: 90% training, 10% testing (used with larger datasets)

**Three-way Splits (Training/Validation/Test):**<br>
When including a validation set for hyperparameter tuning:

- 60/20/20 split: 60% training, 20% validation, 20% testing
- 70/15/15 split: 70% training, 15% validation, 15% testing
- 80/10/10 split: 80% training, 10% validation, 10% testing
> **Terminology**<br>
> Validation set is sometimes called Cross-validation set, Development set, or Dev set.
>
> The term _hold-out set_ is often used to refer to the dataset that is not used for training. Two-way split has one hold-out set (test set), while three-way split has two hold-out sets (validation and test sets).

**Three-way Split is Superior to Two-way Split**<br>

When using a two-way split, we have training and test sets. Let's say we want to train 10 different models and pick the best one.

Let's say our models are neural networks with different architectures. We define a new parameter called **nn** that simply an integer referencing the model. We have 10 different models with different architectures, and we want to pick the best one.
- nn=1: 3 hidden layers, 100 neurons each, ReLU and Adam, etc
- nn=2: 5 hidden layers, 50 neurons each, Tanh and SGD, etc
- ...
- nn=10: 2 hidden layers, 200 neurons each, Leaky ReLU and Adam, etc

After training all these models, we run the test set to get the performance of each model. After all training and comparison, we pick nn=5 as the best model.

Now we have a problem! We have trained all models (nn=1 to 10) on the training set and evaluated them on the test set. But we have used the test set to select the best model. This means that we have used the test set to tune our **new** hyperparameter of the model, **nn**. In other words, we can think of this as we trained a new function that maps the parameter **nn** to the output performance (accuracy, precision, etc) of the model. This is not a good practice because we have used the test set to tune the model, and now we don't know how well this model will perform on unseen data.

In this case, we say that result of test set is **overly optimistic**. This means that real performance of the model nn=5 on unseen data is probably lower than what we got from the test set.

The solve this problem, we use a three-way split. We split the data into training, validation, and test sets. We train all the models on the training set and evaluate them on the validation set to pick the best model. Then we evaluate the best model on the test set to estimate the performance on unseen data. This way, we have not used the test set to tune our final model, and because we haven't made any decisions based on the test set, that ensures that the test set is a fair representation of the unseen data and not overly optimistic estimate of the model performance on generalization of the unseen data.

> We ideally want to avoid making decisions based on the test set during the model tuning and selection process. When we come up with the final model, then at the **very end** we estimate the generalization performance using the test set to fairly estimate the performance of the model on unseen data. In other words, by reserving a separate test set, you insulate your final evaluation from all tuning decisions. Hence, the three-way split is superior to the two-way split.
>
> Note: In model selection, we evaluate those model against the **validation set** not the **test set** to pick the best model.


**Benchmark Datasets**<br>
Benchmark datasets like MNIST often come with predefined splits (MNIST is split approximately 86/14 with 60,000 training and 10,000 test images).

[Cross-validation](ai_machine_learning_cross_validation.md) techniques modify these splits by rotating which portion is used for validation Very large datasets might use smaller test portions (e.g., 95/2.5/2.5) as even small percentages provide enough examples The optimal split depends on dataset size, model complexity, and the specific problem being solved.


## Choosing the Right Metrics
**Classification:** Accuracy, Precision, Recall, F1 Score, ROC-AUC

**Regression:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R^2

> Note: In evaluating the model performance using error metrics, we don't use the regularization term. The regularization term is used to prevent overfitting during the training process, but when we evaluate the model performance, we only use the loss function (e.g. MSE) without the regularization term.

**Clustering:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index

**Time Series:** Mean Absolute Error (MAEP), Mean Squared Error (MSE). Root Mean Squared Error (RMSE),
Mean Absolute Percentage Error (MAPE), Symmetric Mean Absolute Percentage Error (SMAPE)

**Domain-Specific Considerations:** Some domains may require specific evaluation strategies or metrics.
**Cost-Sensitive Evaluation:** In some applications, the cost of false positives and false negatives can be different. Adjust your evaluation metrics to reflect these costs.

**Tips:**
- Metrics like MAE, RMSE, and MAPE provide a standardized way of measuring error across different models. They quantify how far off predictions are from actual values, regardless of the underlying model. These metrics allow for an objective comparison of model performance by providing a single, quantifiable measure of error. This facilitates the evaluation of which model performs better on a specific task.

- In scenarios with highly imbalanced data, precision, recall, and F1 score might be more informative for classification tasks because they provide a clearer picture of how well the model performs on the minority class

- Probabilistic Models: For models that output probabilities (e.g., some classifiers), metrics like log loss or AUC-ROC might provide additional insights into model performance beyond what MAE or RMSE can offer.

## Baseline Model
Establish a simple baseline model to compare against more complex models. This can help in understanding the minimum performance expected and assess the complexity needed.


## Cross Validation
**Cross-Validation** is a technique used to evaluate the performance of a model by training and testing it on multiple subsets of the data.  For details on Cross-Validation, see [Cross-validation](ai_machine_learning_cross_validation.md)

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
