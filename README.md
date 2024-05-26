# Breast Cancer Detection with CART (Classification and Regression Trees)

This project aims to build a machine learning model for breast cancer diagnosis using the CART algorithm. The dataset contains features used for breast cancer diagnosis.

## 1. Dataset

The dataset can be accessed [here](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original). It includes features used for breast cancer diagnosis (e.g., cell nucleus features) along with the class of each instance (benign or malignant).

## 2. Model

In this project, a classification model is built using the CART algorithm. CART (Classification and Regression Trees) learns the relationship between features and classes in the dataset to classify new instances.

## 3. Implementation

Key steps of the project include:

1. *Loading and cleaning the dataset*: Missing values are removed to ensure data quality.
2. *Data augmentation for imbalanced dataset*: Oversampling the minority class to balance the dataset.
3. *Standardizing features*: Features are standardized using StandardScaler.
4. *Splitting into training and testing sets*: The dataset is split into training and testing sets.
5. *Using GridSearchCV for hyperparameter optimization*: Grid search is used to find the best hyperparameters for the model.
6. *Selecting the best model and evaluating on the test set*: The best model is selected based on cross-validation results and evaluated on the test set.
7. *Visualizing the decision tree*: The decision tree is visualized to understand the decision-making process.
8. *Calculating and visualizing performance metrics*: Metrics such as accuracy, F1 score, precision, and recall are calculated and visualized.

## 4. Results

The performance metrics obtained at the end of the project are as follows:

- *Accuracy*: [Accuracy Score]
- *F1 Score*: [F1 Score]
- *Precision*: [Precision Score]
- *Recall*: [Recall Score]

Feel free to explore the project details and code in our repository.
