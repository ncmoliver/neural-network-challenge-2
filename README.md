# neural-network-challenge-2

# Employee Attrition and Department Prediction Model

## Project Overview

This project uses a dense neural network with branching architecture to predict two key outcomes for employees: (1) their department and (2) whether they will stay with the company (attrition). The model utilizes separate outputs to handle multi-class and binary classification with softmax and sigmoid activation functions, respectively.

### Model Performance Summary

- **Department Prediction Accuracy**: 80%
- **Attrition Prediction Accuracy**: 49%

Given the potential class imbalance in the attrition data, accuracy alone may not fully represent the model's predictive power. Thus, we suggest further evaluation using additional metrics such as precision, recall, F1 score, and ROC-AUC.

## Data Preprocessing

- **Scaling and Encoding**: Applied `StandardScaler()` scaling and `OneHotEncoder()` encoding to the dataset to ensure consistent input ranges and categorical handling.
- **Train-Test Splits**: Split the data into training and test sets, scaling and encoding them separately to prevent data leakage.

## Model Architecture

- **Branching Neural Network**: The model consists of two output branches:
  - **Attrition Prediction Branch**: Uses a sigmoid activation function to handle binary classification.
  - **Department Prediction Branch**: Uses a softmax activation function for multi-class classification across department types.

### Model Evaluation

Currently, accuracy is the main metric used, though this may not fully capture model performance due to potential class imbalance. To address this, we recommend further evaluation using:

- Confusion Matrix
- Precision and Recall
- F1 Score
- ROC-AUC for attrition
- Cross-Validation scores to check generalization

## Installation and Requirements

- Python >= 3.7
- TensorFlow >= 2.0
- scikit-learn
- pandas

# References

[Xpert Learning Assistant](https://bootcampspot.instructure.com/courses/6028/external_tools/313)
