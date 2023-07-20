# credit-risk-classification

## Overview of the Analysis

The purpose of this analysis is to be able to classify whether a loan is a "high-risk" or "healthy."

The data used in the model was the loan size, interest rate, the borrower's income, their debt to income, number of accounts, the number of derogatory marks, and finally their total debt.

We were trying to predict whether the status of the loan which would be either 0 or 1. A 0 indicated that the loan was healthy and a 1 indicated that the model was at high risk of defaulting.

A logistic regression model was used for the classification. Before the data was provided into the model we first created our X and y variables. Our y variable is the ```loan_status``` column.
Our X variable was all other columns except for the ```loan_status``` column.

Afterwards, we took the data and split it into different variables to allow for testing of the model's accuracy. It was done using ```train_test_split``` with the following code:

```python
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)
```

We then fit the model with the training data. Next, we used that model to predict some values using the ```X_valid``` data and analyzed its accuracy with the ```y_valid``` data.

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

## Results

* Precision:
  * This measures the accuracy of positive predictions of the model.
    * On label 0, the precision was 1.00, meaning that when the model predicts that a loan is "healthy" it is correct 100% of the time.
    * On label 1, the precision was 0.85, meaning that when a model predicts a loan as "high-risk" it is correct only 85% the time.



* Recall:
  * This measures the ability of the model to identify instances of a specific class correctly
    * On label 0, the recall was 0.99, which means that the model correctly identified 99% of the "healthy" loans.
    * On label 1, the recall was 0.91, which means that the model correctly identified 91% of the "high-risk" loans.



* Accuracy:
  * The accuracy score of the model was 0.99, which means that the model was accurate in its predictions 99% of the time

## Summary

This machine learning model performs very well with predicting whether a loan is "healthy" or at "high-risk".
The model shows high precision and recall for both labels, correctly identifying "healthy" loans with a 100% precision and a 99% recall
and it identifies "high-risk" loans with a 85% precision and a 91% recall. This means
that whenever the model predicts the outcome as "healthy", it is very likely to be accurate and it
also identifies a high portion of "high-risk" loans.

The models accuracy is 99% which means it makes a correct prediction for almost all instances. The model was also
trained and evaluated on a large dataset. The evaluation was done on over 19,000 datapoints.

I would recommend using this ```LogisticRegression``` model for loan predictions. It has a high
precision, recall, and accuracy while being evaluated on a large dataset.