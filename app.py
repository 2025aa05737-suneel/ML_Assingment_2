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
# 2. File Uploader
uploaded_file = st.file_uploader("Choose a CSV file (e.g., test_samples.csv)", type="csv")

if uploaded_file:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df_test.head())

    # Load Model and Scaler
    formatted_name = bc_{model_choice.lower()}.replace(' ', '_')
    model_path = f"model/{formatted_name}.pkl" 
    #model_path = f"model/{model_choice.lower().replace(' ', '_')}.pkl"
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))

    if st.button("Run Prediction"):
        # Prepare data (drop diagnosis if present)
        features = df_test.drop('diagnosis', axis=1) if 'diagnosis' in df_test.columns else df_test
        
        # Scale and Predict
        scaled_features = scaler.transform(features)
        preds = model.predict(scaled_features)
        
        # Display Results
        df_test['Prediction'] = ["Malignant" if p == 0 else "Benign" for p in preds]
        st.success(f"Predictions completed using {bc_model_choice}")
        st.dataframe(df_test)

        # 3. Visualization: Confusion Matrix (Required for Marks)
        if 'diagnosis' in df_test.columns:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(df_test['diagnosis'], preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
