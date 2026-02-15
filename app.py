import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(page_title="Breast Cancer Diagnostic Tool", layout="wide")

st.title("🩺 Breast Cancer Classification Dashboard")
st.write("Upload a CSV file to test the models and see predictions.")
st.sidebar.divider()

# 1. Sidebar for Model Selection
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select Model", 
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost")
)
st.subheader(f"📊 {model_choice} Performance Metrics")

st.sidebar.subheader("Test Data file")

# Load the file to provide it as a download
try:
    # Ensure test_data.csv is in your GitHub root or model folder
    sample_data = pd.read_csv('Data/test_data.csv') 
    
    # Convert dataframe to CSV for the download button
    csv_download = sample_data.to_csv(index=False).encode('utf-8')

    st.sidebar.download_button(
        label="📥 Download Test file",
        data=csv_download,
        file_name='test_data_for_testing.csv',
        mime='text/csv',
        help="Download this file and upload it below to test the model."
    )
except Exception as e:
    st.sidebar.error("test_data.csv not found in repository.")

metrics_df = pd.read_csv('model/comparison_metrics.csv')
if metrics_df is not None:
    # Filter the table for the selected model
    model_stats = metrics_df[metrics_df['ML Model Name'] == model_choice].iloc[0]
    
    # Display metrics in a clean row of columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{model_stats['Accuracy']:.3%}")
    col2.metric("MCC", f"{model_stats['MCC']:.3f}")
    col3.metric("AUC", f"{model_stats['AUC']:.3f}")
    col4.metric("Precision", f"{model_stats['Precision']:.3%}")
    col5.metric("Recall", f"{model_stats['Recall']:.3%}")
    col5.metric("F1", f"{model_stats['F1']:.3%}")
else:
    st.warning("Metrics file not found. Please ensure 'comparison_metrics.csv' is in the model folder.")
# 2. File Uploader
uploaded_file = st.file_uploader("Choose a CSV file (e.g., test_samples.csv)", type="csv")

if uploaded_file:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df_test.head())

    # Load Model and Scaler
 
    model_path = f"model/{model_choice.lower().replace(' ', '_')}.pkl"
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
        st.success(f"Predictions completed using {model_choice}")
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
