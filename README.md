# Customer Churn Predictive Modeling

## What This Project Demonstrates

This project demonstrates an end-to-end predictive modeling workflow for identifying customers at risk of churn.

The analysis moves beyond simply comparing model accuracy. It evaluates how effectively different machine learning algorithms identify customers who actually churn, how well the models generalize to unseen data, and how predictive results could support customer retention decisions.

### Skills Demonstrated

- Exploratory Data Analysis
- Data preprocessing and feature preparation
- Binary classification
- Logistic Regression
- Support Vector Machines
- Random Forest
- Decision Trees
- Model evaluation and comparison
- Confusion matrix analysis
- ROC-AUC analysis
- Cross-validation
- Feature importance analysis
- Overfitting and generalization analysis
- Business-focused model selection
- Translating predictive output into business recommendations

### Tools & Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Google Colab
- GitHub

---

## Business Problem

Customer churn can reduce recurring revenue, increase customer acquisition costs, and weaken long-term customer relationships.

A retail bank wants to identify customers who are more likely to leave so retention resources can be focused on customers with elevated churn risk.

### Business Question

Can customer account and behavioral characteristics be used to predict which customers are most likely to churn?

### Modeling Approach

Four classification models were developed and compared:

1. Logistic Regression
2. Support Vector Machine
3. Random Forest
4. Decision Tree

Model selection considered multiple performance dimensions rather than relying on accuracy alone.

---

## Dataset Overview

The analysis uses a retail banking customer churn dataset containing 10,000 customer records.

The modeling target is `Exited`:

- `0` = Customer retained
- `1` = Customer churned

### Key Dataset Characteristics

- Total customers: 10,000
- Retained customers: 7,963
- Churned customers: 2,037
- Churn rate: 20.37%
- Retention rate: 79.63%

The class distribution creates an imbalanced classification problem. Because most customers remain with the bank, accuracy alone is not sufficient for evaluating model performance.

### Customer Churn Distribution

![Customer Churn Distribution](images/churn_distribution.png)

### Predictive Features

After removing identifier fields, nine customer characteristics were used for modeling:

- Credit Score
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Active Member Status
- Estimated Salary

Identifier fields including `RowNumber`, `CustomerId`, and `Surname` were excluded because they function as identifiers rather than meaningful predictive customer characteristics.

See `data/README.md` for additional dataset information.

---

## Exploratory Data Analysis

Before training the models, I examined relationships between the numerical features and the churn target.

### Feature Correlation Matrix

![Feature Correlation Matrix](images/correlation_matrix.png)

The correlation matrix provided an initial view of linear relationships in the dataset.

### Initial Findings

Several features showed relationships with churn, but the exploratory analysis also demonstrated an important modeling principle: correlation alone does not determine predictive value.

Some variables with relatively weak linear relationships later became important within nonlinear tree-based models.

For this reason, the correlation analysis was used as an exploratory tool rather than as the sole method for selecting predictive features.

---

## Data Preparation

The dataset was prepared for machine learning before model development.

### Preprocessing Steps

- Removed identifier fields that were not useful for prediction
- Encoded the categorical `Gender` variable numerically
- Separated predictive features from the `Exited` target
- Created an 80/20 stratified train-test split
- Preserved the churn distribution across training and testing datasets
- Standardized numerical features using `StandardScaler`
- Fitted the scaler only on the training data to prevent information leakage

### Train-Test Split

- Training set: 8,000 customers
- Testing set: 2,000 customers
- Approximate churn rate in both sets: 20.4%

### Modeling Pipeline

`Raw Customer Data` → `Data Cleaning` → `Feature Encoding` → `Train/Test Split` → `Feature Scaling` → `Model Training` → `Model Evaluation`

---

## Model Development & Performance Comparison

Four classification algorithms were trained using the same training and testing framework.

### Performance Results

| Model | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 80.50% | 58.25% | 14.74% | 0.2353 | 0.7711 | 80.74% |
| SVM | 85.80% | 83.61% | 37.59% | 0.5186 | 0.7976 | 85.06% |
| Random Forest | 85.95% | 80.00% | 41.28% | 0.5446 | 0.8472 | 85.71% |
| Decision Tree | 83.30% | 64.20% | 40.54% | 0.4970 | 0.7788 | 82.85% |

### Visual Performance Comparison

![Model Accuracy and Precision Comparison](images/model_accuracy_precision.png)

![Model Recall and F1-Score Comparison](images/model_recall_f1.png)

### What the Results Show

Random Forest achieved the highest testing accuracy at 85.95%.

SVM achieved the highest precision at 83.61%, meaning that when it predicted churn, those predictions were more likely to be correct.

Random Forest achieved the highest recall at 41.28% and the highest F1-score at 0.5446, providing a stronger balance between identifying churners and maintaining precision.

Logistic Regression demonstrated why accuracy alone can be misleading in an imbalanced classification problem. Although the model achieved 80.50% testing accuracy, it identified only 14.74% of customers who actually churned.

---

## Confusion Matrix Analysis

Confusion matrices were used to examine the types of classification errors produced by each model.

### Logistic Regression vs. Support Vector Machine

![Logistic Regression and SVM Confusion Matrices](images/confusion_matrix_lr_svm.png)

### Random Forest vs. Decision Tree

![Random Forest and Decision Tree Confusion Matrices](images/confusion_matrix_rf_dt.png)

### Churn Classification Results

| Model | Actual Churners Identified | Churners Missed | False Churn Predictions |
|---|---:|---:|---:|
| Logistic Regression | 60 | 347 | 43 |
| SVM | 153 | 254 | 30 |
| Random Forest | 168 | 239 | 42 |
| Decision Tree | 165 | 242 | 92 |

### Business Interpretation

Random Forest identified the largest number of actual churners, correctly detecting 168 of the 407 churned customers in the test set.

SVM generated the fewest false-positive churn predictions, which explains its higher precision.

Decision Tree identified nearly as many churners as Random Forest but generated substantially more false positives.

Logistic Regression missed 347 of the 407 churners, showing that its relatively high accuracy was driven primarily by correctly identifying the majority retained class.

For a churn problem, these tradeoffs matter because missing a customer who is likely to leave may represent a lost opportunity for retention.

---

## ROC-AUC & Model Discrimination

ROC-AUC measures how effectively a model distinguishes between retained and churned customers across classification thresholds.

![ROC Curves](images/roc_curves.png)

### ROC-AUC Results

| Model | ROC-AUC |
|---|---:|
| Random Forest | 0.8472 |
| SVM | 0.7976 |
| Decision Tree | 0.7788 |
| Logistic Regression | 0.7711 |

Random Forest achieved the highest ROC-AUC at 0.8472, providing the strongest overall discrimination between customers who remained and customers who churned.

This result supports the broader model comparison because Random Forest performed well across several evaluation measures rather than dominating only one metric.

---

## Feature Importance & Business Insights

Tree-based models provide feature-importance estimates that help identify which variables contributed most strongly to their predictions.

### Tree-Based Feature Importance

![Random Forest and Decision Tree Feature Importance](images/feature_importance.png)

### Random Forest Feature Importance

| Feature | Importance |
|---|---:|
| Age | 0.3387 |
| Number of Products | 0.2283 |
| Balance | 0.1029 |
| Estimated Salary | 0.0938 |
| Credit Score | 0.0901 |
| Active Member Status | 0.0722 |
| Tenure | 0.0446 |
| Gender | 0.0179 |
| Has Credit Card | 0.0110 |

### Key Findings

`Age` was the most influential feature in the Random Forest model, accounting for approximately 33.9% of total feature importance.

`NumOfProducts` ranked second at approximately 22.8%.

Together, these two variables represented approximately 57% of the model's total feature importance.

Other meaningful contributors included customer balance, estimated salary, credit score, and active membership status.

### An Interesting Modeling Insight

One of the more useful findings emerged when comparing exploratory correlation analysis with Random Forest feature importance.

`NumOfProducts` showed only a weak linear correlation with churn during exploratory analysis, yet it became the second-most-important feature in the Random Forest model.

This suggests that the relationship between product usage and churn may be nonlinear or may depend on interactions with other customer characteristics.

The finding demonstrates why predictive modeling can uncover relationships that are not obvious from correlation analysis alone.

Feature importance represents predictive contribution within the model. It does not establish that these variables cause customers to churn.

### From Model Output to Business Action

The feature-importance results could guide further customer analysis.

For example, the business could examine whether churn risk changes across:

- Customer age groups
- Number of banking products held
- Account balance levels
- Active versus inactive customers
- Credit score ranges

These relationships would require further analysis before being translated into retention strategies.

---

## Model Generalization & Overfitting

A strong predictive model should perform well on both training data and previously unseen test data.

Training and testing accuracy were compared to evaluate generalization.

![Training vs Testing Accuracy](images/train_test_comparison.png)

### Training vs. Testing Performance

| Model | Training Accuracy | Testing Accuracy | Gap |
|---|---:|---:|---:|
| Logistic Regression | 80.73% | 80.50% | 0.22% |
| SVM | 85.96% | 85.80% | 0.16% |
| Random Forest | 89.38% | 85.95% | 3.43% |
| Decision Tree | 89.41% | 83.30% | 6.11% |

### Generalization Findings

Logistic Regression and SVM showed very small differences between training and testing accuracy, indicating consistent performance across the two datasets.

Random Forest produced a 3.43 percentage-point train-test gap. Although larger than the gaps for Logistic Regression and SVM, the model maintained strong testing performance.

Decision Tree showed the largest gap at 6.11 percentage points, indicating greater evidence of overfitting.

The comparison between Random Forest and Decision Tree is particularly useful. Both achieved approximately 89.4% training accuracy, but Random Forest maintained 85.95% testing accuracy compared with 83.30% for Decision Tree.

This indicates that the Random Forest ensemble generalized more effectively than the standalone Decision Tree.

---

## Cross-Validation Analysis

Five-fold cross-validation was used as an additional measure of model stability.

Rather than relying entirely on a single train-test split, cross-validation evaluates model performance across multiple subsets of the training data.

![Cross Validation Performance](images/cross_validation.png)

### Cross-Validation Results

| Model | CV Mean Accuracy | Test Accuracy | Difference |
|---|---:|---:|---:|
| Logistic Regression | 80.74% | 80.50% | 0.24% |
| SVM | 85.06% | 85.80% | 0.74% |
| Random Forest | 85.71% | 85.95% | 0.24% |
| Decision Tree | 82.85% | 83.30% | 0.45% |

### Cross-Validation Findings

Random Forest achieved the highest mean cross-validation accuracy at 85.71%.

Its cross-validation accuracy was also close to its 85.95% held-out test accuracy, providing evidence of stable performance across different subsets of the data.

SVM also performed consistently, achieving 85.06% mean cross-validation accuracy and 85.80% testing accuracy.

All four models produced cross-validation results relatively close to their held-out testing performance.

---

## Final Model Ranking

The final model was selected using multiple evaluation criteria rather than relying on a single performance metric.

The selection framework considered:

- Testing accuracy
- F1-score
- ROC-AUC
- Cross-validation performance
- Generalization

### Custom Selection Framework

A project-specific weighted Selection Score was calculated using:

| Evaluation Criterion | Weight |
|---|---:|
| Testing Accuracy | 30% |
| F1-Score | 25% |
| ROC-AUC | 25% |
| Cross-Validation Accuracy | 15% |
| Generalization | 5% |

The Selection Score is a custom decision metric created for this analysis. It is not a standard machine learning evaluation metric and should be interpreted alongside the individual model-performance measures.

### Final Ranking

| Rank | Model | Selection Score | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Random Forest | 0.7826 | 85.95% | 80.00% | 41.28% | 0.5446 | 0.8472 |
| 2 | SVM | 0.7640 | 85.80% | 83.61% | 37.59% | 0.5186 | 0.7976 |
| 3 | Decision Tree | 0.7401 | 83.30% | 64.20% | 40.54% | 0.4970 | 0.7788 |
| 4 | Logistic Regression | 0.6641 | 80.50% | 58.25% | 14.74% | 0.2353 | 0.7711 |

### Recommended Model: Random Forest

Random Forest was selected because it provided the strongest overall balance across the evaluation criteria.

While SVM achieved the highest precision at 83.61%, Random Forest achieved the highest testing accuracy, recall, F1-score, ROC-AUC, and cross-validation accuracy.

### Final Model Performance

| Metric | Random Forest |
|---|---:|
| Training Accuracy | 89.38% |
| Testing Accuracy | 85.95% |
| Precision | 80.00% |
| Recall | 41.28% |
| F1-Score | 0.5446 |
| ROC-AUC | 0.8472 |
| 5-Fold CV Accuracy | 85.71% |
| Train-Test Gap | 3.43% |
| Selection Score | 0.7826 |

The model's performance across multiple evaluation measures made Random Forest the strongest candidate for the customer churn prediction objective.

---

## Business Recommendation

The Random Forest model could serve as the foundation for a customer churn risk-scoring process.

Rather than treating every customer as having the same retention risk, the bank could use predicted churn probabilities to prioritize customers for further analysis and targeted intervention.

### Potential Decision Workflow

`Customer Data` → `Random Forest Model` → `Churn Risk Score` → `High-Risk Customer Segment` → `Targeted Retention Action` → `Measure Outcome`

Potential retention actions could include:

- Prioritized outreach to customers with elevated churn risk
- Review of product relationships for high-risk customers
- Personalized retention offers
- Customer service follow-up
- Analysis of account engagement and activity
- Testing different retention strategies across risk segments

The model should support business decisions rather than automatically determine how a customer is treated.

---

## Key Takeaways

This project demonstrated that selecting a predictive model requires more than comparing accuracy scores.

Three findings stood out:

1. Accuracy can hide poor minority-class performance. Logistic Regression achieved 80.50% accuracy but identified only 14.74% of actual churners.

2. Model tradeoffs matter. SVM produced the highest precision, while Random Forest provided the strongest balance of recall, F1-score, ROC-AUC, and overall accuracy.

3. Model reliability matters alongside performance. Random Forest's cross-validation accuracy of 85.71% closely matched its 85.95% testing accuracy, providing evidence of stable performance across different subsets of the data.

The final result was not simply a higher-performing classification model. The analysis produced a framework for identifying churn risk and translating predictive output into customer retention decisions.

---

## Project Structure

```text
customer-churn-predictive-modeling/
│
├── data/
│   └── README.md
│
├── images/
│   ├── churn_distribution.png
│   ├── correlation_matrix.png
│   ├── model_accuracy_precision.png
│   ├── model_recall_f1.png
│   ├── confusion_matrix_lr_svm.png
│   ├── confusion_matrix_rf_dt.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── train_test_comparison.png
│   └── cross_validation.png
│
├── notebooks/
│   └── customer_churn_modeling.ipynb
│
├── .gitignore
└── README.md
```

### Repository Contents

- `data/` — Information about the customer churn dataset used for the analysis
- `images/` — Visualizations generated during exploratory analysis and model evaluation
- `notebooks/` — Complete Python machine learning workflow

---

## Limitations & Future Improvements

The project provides a baseline for customer churn classification, but several areas could be explored before applying the model in a production environment.

### Current Limitations

- The target variable is imbalanced, with approximately 20% of customers belonging to the churn class.
- Random Forest identified 41.28% of actual churners, meaning a substantial portion of churned customers were still missed.
- The models were evaluated using a single dataset rather than customer data collected across multiple time periods.
- Feature importance identifies variables used by the model but does not establish causal relationships with churn.
- The current analysis uses a default classification threshold rather than optimizing the threshold around the business cost of false positives and false negatives.
- The analysis evaluates predictive performance but does not measure the financial impact of retention interventions.

### Future Improvements

- Tune Random Forest hyperparameters using `GridSearchCV` or `RandomizedSearchCV`
- Evaluate class-weighted models to address class imbalance
- Compare additional ensemble and boosting algorithms
- Add Precision-Recall curves for minority-class evaluation
- Optimize the classification threshold based on business objectives
- Use SHAP values to improve individual prediction explainability
- Convert predicted probabilities into customer risk segments
- Incorporate customer lifetime value into retention prioritization
- Estimate the financial value of successful churn interventions
- Monitor model performance and data drift over time

A future version could extend the analysis from predicting churn to determining which high-risk customers should receive retention offers based on expected customer value and intervention cost.

---

## Author

Shirley Landon

Data Analytics | Predictive Modeling | Python | SQL | Machine Learning | Business Analytics

This project demonstrates my approach to translating a business problem into a structured predictive modeling workflow, evaluating competing machine learning models, and communicating the results in a way that supports business decision-making.
