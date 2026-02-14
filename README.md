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

Dataset Description
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
Testing Split: 28 rows were reserved for Streamlit testing to validate real-time model predictions.

Models used: Comparison Table
The following table summarizes the evaluation metrics for all 6 models trained:

ML Model Name,Accuracy,AUC,Precision,Recall,F1,MCC
Logistic Regression,0.944954,0.976727,0.923077,1.000000,0.960000,0.879425
Decision Tree,0.917431,0.878378,0.888889,1.000000,0.941176,0.820166
kNN,0.944954,0.953641,0.923077,1.000000,0.960000,0.879425
Naive Bayes,0.899083,0.977290,0.886076,0.972222,0.927152,0.772873
Random Forest,0.917431,0.972785,0.888889,1.000000,0.941176,0.820166
XGBoost,0.935780,0.957958,0.911392,1.000000,0.953642,0.859632
