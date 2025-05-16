---
date: "2024-10-03"
draft: false
title: "Machine Learning Development Process"
description: "An overview of the machine learning development process as an iterative workflow of data engineering, model training, and deployment."
tags:
    - "AI"
---
Machine Learning development is an [empirical process](https://www.deeplearning.ai/the-batch/iteration-in-ai-development/). So, when you start, don’t overthink and spend too much time on the design of the system, types of algorithms, model architecture, hyperparameters, and so on. Build the first version quick, get the feedback, analyze the result and improve your system iteratively.

The common steps of a machine learning project:
1. **Data Engineering**: Define, collect and preprocess data.
2. **Exploratory Data Analysis (EDA) and Feature Engineering**: Data Analysis, cleansing and visualization of data to understand its structure, relations, patterns, and potential issues. Then in an iterative process, transform and select data representations to best capture information relevant for model training and performance.
3. **Modeling**: Training, Evaluation, Improvement and Tuning, and iterate until the model's performance is satisfactory.
4. **Operation**: Deployment, Monitor the ongoing predictions, manage models and versions (artifacts organization), feedback loops and continuous learning/retraining, scaling, MLOps, etc.

Although as we discussed, machine learning development is an interative workflow between the above steps as shown in the following diagram:

![](images/ml_development_process.svg)


In the above diagram:

- Developing AI and ML solutions is not a linear but an **iterative** one. It's like a loop, and you may need to go back and forth between the steps until you get the desired results.

- The above diagram shows **Feature Engineering** as a separate step, but in the real world, it's an ongoing activity during the EDA process. You explore, clean, and engineer features iteratively until you are satisfied with the read-to-training dataset.

- The iterative loop of the **Modeling** step is core cycle of this process. However, in many cases, you may need to go back to the **Feature Engineering** step or even first step (i.e. Data Engineering) to collect more data or improve the quality of the data.

This [article by MLflow](https://mlflow.org/docs/latest/introduction/index.html) also illustrates the iterative nature of the machine learning development process as we discussed above.


### Data Engineering
See [Data Engineering](data_engineering.md) for more details.

### Exploratory Data Analysis (EDA) and Feature Engineering

#### Visualization Techniques
Visualzation techniques are key tools for EDA (Cleaning and Feature Engineering). So, this is not a separate step, but it's an ongoing activity during the EDA process.

Visualization is like a guide for you to help you to understand the data (stats, outliers, patterns, etc), and to make decisions about before and after data cleaning and feature engineering.

We use the following visualization techniques:

- **Histograms**: To show the distribution of a single variable.
- **Scatter Plots**: To show the relationship between two variables (feature vs feature, and feature vs target).
- **Pairplot**: To show the relationship between multiple variables. Similar to Scatter Plots, but it shows all possible combinations of variables.
- **Heatmaps**: To show the correlation between variables. To detect multi-collinearity between features. i.e. when two features move together (positive or negative correlation).
- **Box Plots**: To show the distribution of a single variable. It shows the median, quartiles, and outliers of the variable.

For seeing these visualizations in action, go to [Linear Regression using Scikit-Learn](labs/linear_regression_scikit_learn.ipynb).


#### Feature Engineering
Feature engineering is the process of selecting the right features (feature reduction), creating new features from the existing features, and transforming the existing ones to new ones. It's a crucial step in the machine learning process, as it can significantly impact the performance of the model.

In feature engineering to verify our hypothesis, we use the visualization techniques (discussed above) to make the right decisions.

See further details here at [Feature Engineering](feature_engineering_machine_learning.md).

### Modeling

Modeling is by itself an iterative process which cycle through the following steps:
- **Model Architecture and Data Setup**: Define the model algorithms, architecture, hyperparameters, and data setup.
- **Model Training**: Train the model on the training dataset.
- [**Model Evaluation**](evaluation_metrics_machine_learning.md): Evaluate the model on the validation dataset, diagnostic of bias and variance, and error analysis.

![](images/modeling_iteration_process.svg)

The result of the evaluation is used to decide whether to continue with the current model or to go back to the first step (i.e. modify the model architecture and data setup) and repeat the process.

> Modeling is an iterative process. Almost always, the first model won't be the best one. So, you need to iterate and improve the model until you get the desired results.

### Operation
This stage is combination of sub-stages which overall aims to bring and maintain the model in productin with the best possible performance, operations and costs. MLOps is a term which is predominantly used to describe this stage. However one can argue that MLOps is an end to end process and starts from the data engineering stage.

In this stage, we focus on the following areas:
- **Deployment**: Deployment of the model to production in a automated and efficient way. Automation, CI/CD, and infrastructure as code (IaC) are key concepts in this stage.
- **Monitoring and Management**: This includes monitoring the model's performance, managing the model versions, auto scaling the model, optimizing the operations and costs. Automation of the deployment and monitoring process. Re-training the model when needed
- **Feedback Loop**: This is a crucial part of the MLOps process. It provides the feedback of the model's performance in production to the data engineering and modeling stages, which then help the newer and better versions of the model to be developed. This is a continuous process and should be automated as much as possible.
