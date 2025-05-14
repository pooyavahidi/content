---
date: "2025-01-20"
draft: false
title: "Cross Validation in Machine Learning"
description: "An overview of cross-validation techniques in machine learning including Hold-out, K-Fold, Stratified K-Fold, and Time Series Split."
tags:
    - "AI"
---
Cross-validation techniques are used to evaluate the performance of models by splitting the dataset into multiple segments, training the model on some segments, and testing it on others. The goal of Cross-validation is to evaluate the performance of different models across multiple splits of the dataset, providing a more robust assessment of the model's generalization capabilities.

We use cross-validation to:
- Compare algorithms: You can use cross-validation to compare the performance of different modeling algorithms on the same dataset to select the most appropriate one for your problem. (e.g. Hold-out validation)

- Tune the hyperparameters: Cross-validation can be used to compare models trained with different sets of hyperparameters to find the optimal configuration for a given algorithm. (e.g. Hold-out validation)

- Compare the performance of a model versions (with the same algorithm and hyperparameters) on different splits (usually K-Fold for non-sequential and Walk-Forward for timeseries).


## Types of Cross-Validation
Here is the list of common methods to perform cross-validation:

**Non-sequential Data:**
- Hold-out Vallidation (Two-way or Three-way Split)
- K-Fold Cross Validation.
- Leave-One-Out Cross Validation.
- Stratified K-Fold Cross Validation.

**Sequential Data (Time Series):**
- Time Series Split (or Sequntial Split)
- Walk-Forward Validation
- Blocked Cross-Validation

> Note: The way of splitting the dataset is very differnt for time series due to the sequential nature of the data. In time series, the order of observations is important, and using future data to predict past or current states can lead to unrealistic or misleading model evaluations, known as "look-ahead bias."


## Feature Selection and Cross-Validation
Feature selection often is a prior step to the cross-validation in the model development workflow, serving to initially reduce the feature space using domain knowledge or statistical methods. However, the process is iterative: cross-validation evaluates model performance with the selected features, informing adjustments to the feature set. This cycle continues—adjusting features based on cross-validation feedback—until an optimal balance of model performance and complexity is achieved. Thus, while feature selection initially sets the stage, both processes are closely intertwined, each informing and refining the other to enhance the model's predictive accuracy and generalizability.



## Non-sequential Data
For non-sequential data, we can use the following methods to perform cross-validation. Non-sequential data refers to datasets where the order of observations does not carry meaningful information for analysis, predictions, or decision-making processes. Each data point is independent of others, and rearranging the dataset does not affect its interpretation or the outcomes of analytical models applied to it.

Examples are image and audio data, textual data, medical records, real estate data, etc.

### Hold-out Validation
Hold-out is the simplest kind of cross validation and it's simply splits the dataset into two parts: training and testing. The model is trained on the training set and evaluated on the testing set. The performance metric is calculated on the testing set.

In this method, we train multiple models on the same dataset splits and compare their performance using a scoring system (which we need define, could be accuracy, percision, recall, AUC metric, etc). This method is useful when we want to compare different algorithms or hyperparameters, or both.

Hold-out Validation is one of the most common methods of cross-validation. The data is usually divided into two sets (two-way split) or three sets (three-way split):

**Two-way Splits (Training/Test)**<br>
The common splits are:
- 80/20 split: 80% training, 20% testing (common approach)
- 70/30 split: 70% training, 30% testing (used when more evaluation data is needed)
- 90/10 split: 90% training, 10% testing (used with larger datasets)

**Three-way Splits (Training/Validation/Test)**<br>
The common splits are:

- 60/20/20 split: 60% training, 20% validation, 20% testing
- 70/15/15 split: 70% training, 15% validation, 15% testing
- 80/10/10 split: 80% training, 10% validation, 10% testing
- For very large datasets might use smaller test portions (e.g., 95/2.5/2.5) as even small percentages provide enough examples The optimal split depends on dataset size, model complexity, and the specific problem being solved.

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


### K-Fold Cross Validation

In **k-fold cross-validation**, you split your entire dataset into **k** equally (or nearly equally) sized “folds.” Then you run **k** training-and-evaluation experiments (often called “rounds” or “folds”), each time using:

- **k–1 folds** as the **training set**, and
- the **remaining 1 fold** as the **validation set**.

By cycling through all k possible choices of validation fold, you get k performance estimates which you then **average** to get a more reliable measure of your model’s generalization performance (and reduce variance due to any one particular train/test split).

![](images/cross_validation_kfold.svg)

In the above image:
- Each **row** corresponds to one of the k = 5 “rounds” (labeled Model 1…Model 5).
- In each round, the **Test** fold is held out for testing, while the **Train** folds are used to train the model.
- Over the 5 rounds you’ll have 5 evaluation scores; you then **average** them to obtain your final cross-validated performance.

**Available Data**<br>
- Using **two-way split**: The available data is whole of the dataset including training and test sets.
- Using **three-way split**: The available data is the training and validation sets only. We seprate the test set before splitting the data into k folds, and then go through the k-fold cross-validation process. This ensures that the test set is **not** used in any of the tuning or selection process, and is only used at the end to evaluate the final model.

**Averaging the results**<br>
The mean performance metrics across all k iterations. For example, if the 5 models in the image achieved accuracies of 82%, 85%, 80%, 87%, and 84%, the final reported accuracy would be the average: 83.6%. Averaging provides several benefits:

- More reliable estimation: Instead of relying on a single train/validation split (which might be lucky or unlucky), you get k different evaluations.

- Reduced variance: Averaging across multiple validation sets provides a more stable estimate of model performance.

- Efficient use of data: Every data point gets used for both training and validation (though never in the same iteration), making maximum use of limited data.

This technique ensures every data point has been in the validation set exactly once and in the training set k–1 times, giving you a robust estimate of how your model behaves on unseen data.

> In practice you’ll most often see k=5 or k=10 folds cross-validation, with 5-fold in many production or auto-ML pipelines. Although in theory, k can be any positive integer.


### Leave-One-Out Cross Validation
In this method, we split the dataset into *N* subsets, where *N* is the number of samples in the dataset. Each sample is used as the test set once while the remaining samples form the training set. This method is useful when we have a small dataset.

### Stratified K-Fold Cross Validation
It's good method when you have imbalanced dataset. It's similar to K-Fold Cross Validation but the difference is that each fold is made by preserving the percentage of samples for each class.


## Time Series
Traditional cross-validation is not directly applicable to time series data due to the sequential nature of the data. In time series, the order of observations is important. The following methods ensure that the temporal relationships in the data are respected, providing more realistic evaluations of how models will perform in real-world, time-based scenarios.

Example of time series data are stock prices, weather data, sensor data, logs, etc.

> Note: sometime our goal is to predict time-independent events, in such cases we can use non-sequential methods. For example, time-indendent anamoly detection (e.g. temperature exceeding a threshold), or time-independent classification (e.g. spam detection).

### Time Series Split (or Sequential Split)
This method involves sequentially splitting the time series data into training and test sets. For each split, the model is trained on past data and tested on future data, respecting the temporal order of observations. This approach can be iterated multiple times by expanding the training set and moving the test set forward in time.

### Walk-Forward Validation (Rolling cross-validation):
Similar to the time series split but with a moving window for the test set. After each training phase, the model is tested on the next time step(s), then the test set is incorporated into the training set, and the process repeats. This closely mimics real-world scenarios where models are updated as new data becomes available.

This method is also known as *Rolling out-of-sample*. This terminology emphasizes the method's approach of incrementally moving the test window forward in time, continuously retraining the model on a "rolling" basis with the most recent data available. This technique is particularly suited to time series forecasting, where models are **periodically updated** to make predictions for the next time period, ensuring that the evaluation mirrors how the model would be used in practice.

Good read on [Walk-Forward Validation](https://arxiv.org/pdf/2001.09055.pdf)

### Blocked Cross-Validation:
For scenarios where time series can be divided into segments (blocks) that are considered independently (e.g., different seasons or years), this method allows for cross-validation within each block, ensuring that temporal order within each block is maintained.


## Resources
- [Cross Validaiton](https://www.cs.cmu.edu/~schneide/tut5/node42.html)
- [Cross Validation by neptune.ai](https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right)
