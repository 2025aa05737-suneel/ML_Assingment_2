# ML_Assingment_2
Breast Cancer Classification Application
**Live App:** https://2025aa05737-breast-cancer-detector.streamlit.app/

## Problem Statement

The goal of this project is to evaluate and compare the performance of various Machine Learning models on a binary classification task for Breast Cancer detection(using multiple classification algorithms).

 The objectives are:

1. Build and train 6 different classification models on a spam email dataset
2. Evaluate each model using multiple performance metrics (Accuracy, AUC, Precision, Recall, F1 Score, MCC)
3. Create an interactive Streamlit web application to demonstrate the models
4. Deploy the application on Streamlit Community Cloud
5. Compare model performances and provide insights

## Dataset Description**
**Source:** (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

The Spambase dataset is a binary classification dataset containing email messages labeled as spam or legitimate (ham). It consists of emails collected for spam detection research.

**Dataset Characteristics:**
- **Total Instances:** 569
- **Total Features:** 30
- **Target Variable:** Diagnosis (M = malignant, B = benign)
- **Class Distribution:**
  - maligant: 212
  - bengin: 357
- **Missing Values:** None

**Preprocessing:** The models were trained on features scaled and encoded to optimize the performance of both linear and tree-based algorithms.
Testing Split: 90/20

## Repository Structure


```
ML-Assignment_2
├── Data/
│   ├── master_data.csv        # Combined dataset
│   ├── test_data.csv          # Raw test data for Streamlit UI testing
│   └── wdbc.data              # Original UCI dataset file
├── model/
│   ├── 2025aa05737__ML_assignment.ipynb  # Training script Notebook
│   ├── comparison_metrics.csv # Saved metrics for Streamlit display
│   ├── decision_tree.pkl      # Serialized Model Files
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl             # StandardScaler for preprocessing
├── app.py                     # Streamlit Application code
├── README.md                  # Project documentation & observations
└── requirements.txt           # Deployment dependencies
```

**requirements.txt

streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn



Models used: Comparison Table
The following table summarizes the evaluation metrics for all 6 models trained:


## ML Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| **Decision Tree** | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| **kNN** | 0.9474 | 0.9820 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| **Naive Bayes** | 0.9649 | 0.9974 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| **Random Forest (Ensemble)** | 0.9649 | 0.9956 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| **XGBoost (Ensemble)** | 0.9561 | 0.9931 | 0.9583 | 0.9718 | 0.9650 | 0.9064 |

 
### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Best Overall Performer.** It achieved the highest Accuracy (0.9449) and a perfect Recall (1.0). It is the most balanced model for this dataset. |
| **Decision Tree** | While it maintained perfect Recall, it had the lowest AUC (0.8784), suggesting it may struggle with generalizability compared to ensemble methods. |
| **kNN** | Matches Logistic Regression in Accuracy and F1-score (0.96), proving that a distance-based approach is highly effective for this specific data distribution. |
| **Naive Bayes** | Features the highest AUC (0.9773), showing excellent separation power, though it had the lowest Accuracy and MCC among all models. |
| **Random Forest (Ensemble)** | Highly reliable with perfect Recall, but slightly lower Precision (0.8889) than Logistic Regression or XGBoost on this test set. |
| **XGBoost (Ensemble)** | **Strong Runner-up.** It provides a high MCC (0.8596) and perfect Recall, making it a very robust choice for production deployment. |
