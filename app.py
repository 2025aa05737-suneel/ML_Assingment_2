import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="Breast Cancer Diagnostic Tool", layout="wide")

st.title("🩺 Breast Cancer Classification Dashboard")
st.write("Upload a CSV file to test the models and see predictions.")

# 1. Sidebar for Model Selection
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select Model", 
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")
)
