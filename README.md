# Customer Churn Predictive Modeling

## What This Project Demonstrates

This project demonstrates an end-to-end machine learning workflow for predicting customer churn and selecting the classification model that provides the strongest overall performance.

Using a retail banking dataset containing 10,000 customer records, I built and evaluated four supervised machine learning models to determine which approach could best identify customers at risk of leaving the bank.

### Skills Demonstrated

* Exploratory Data Analysis (EDA)
* Data preprocessing and feature engineering
* Binary classification
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* Decision Tree
* Train-test splitting and feature scaling
* Cross-validation
* Model performance evaluation
* Precision, recall, F1-score, and ROC-AUC analysis
* Confusion matrix analysis
* Feature importance analysis
* Overfitting and generalization assessment
* Translating model results into business recommendations

### Tools & Technologies

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `Google Colab`

---

## Business Problem

Customer churn can reduce recurring revenue and increase the cost of acquiring replacement customers. A retail bank wants to identify customers who are at greater risk of leaving so retention efforts can be targeted before those customers exit.

The dataset contains demographic, financial, and account activity information for 10,000 customers. The target variable, `Exited`, identifies whether a customer remained with the bank or churned.

Exploratory analysis identified an overall churn rate of 20.37%, meaning approximately one in five customers in the dataset had left the bank.

### Business Question

Can customer characteristics and account behavior be used to predict customer churn, and which classification model provides the best balance of predictive performance and generalization to unseen customers?

### Modeling Approach

Four classification algorithms were evaluated:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. Decision Tree

Rather than selecting a model based only on accuracy, performance was evaluated across multiple measures including precision, recall, F1-score, ROC-AUC, cross-validation performance, and the difference between training and testing accuracy.

This approach provides a more complete assessment of whether a model can identify customers at risk while continuing to perform reliably on unseen data.

## Dataset Overview

The analysis uses a retail banking customer dataset containing 10,000 customer records and 13 original variables. The target variable is `Exited`, where `0` represents a customer who remained with the bank and `1` represents a customer who churned.

The dataset includes customer demographics, financial information, product relationships, and account activity that can be evaluated as potential predictors of churn.

### Key Dataset Characteristics

| Metric             |  Value |
| ------------------ | -----: |
| Customer Records   | 10,000 |
| Original Variables |     13 |
| Retained Customers |  7,963 |
| Churned Customers  |  2,037 |
| Churn Rate         | 20.37% |
| Missing Values     |      0 |

### Customer Churn Distribution

![Customer Churn Distribution](images/churn_distribution.png)

The target variable is imbalanced. Approximately 79.63% of customers remained with the bank, while 20.37% churned.

This imbalance is important when evaluating model performance. Accuracy alone could make a model appear effective even if it performs poorly at identifying customers who actually churn. For this reason, the models were also evaluated using precision, recall, F1-score, ROC-AUC, and confusion matrices.

### Predictive Features

| Feature           | Description                              |
| ----------------- | ---------------------------------------- |
| `CreditScore`     | Customer credit score                    |
| `Gender`          | Customer gender                          |
| `Age`             | Customer age                             |
| `Tenure`          | Length of customer relationship          |
| `Balance`         | Account balance                          |
| `NumOfProducts`   | Number of bank products used             |
| `HasCrCard`       | Whether the customer has a credit card   |
| `IsActiveMember`  | Whether the customer is an active member |
| `EstimatedSalary` | Estimated customer salary                |

Three identifier fields, `RowNumber`, `CustomerId`, and `Surname`, were removed before modeling because they were not used as predictive features.

After preprocessing, the modeling dataset contained nine predictor variables and the `Exited` target.

