import streamlit as st
import pandas as pd

st.set_page_config(page_title="Student Academic Performance Analysis", layout="wide")

# Sidebar
st.sidebar.title("Student Performance Analyzer")
st.sidebar.markdown("""
**Project Description**

Identify which factors most significantly influence student academic performance using ML models and a public dataset.

[Dataset on Kaggle](https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset)
""")

# Feature selection
feature_options = [
    "hours studied",
    "attendance",
    "sleep hours",
    "extracurricular activities",
    "department",
    "participation score"
]
selected_features = st.sidebar.multiselect("Select Features to Analyze", feature_options, default=feature_options)

# Model selection
tab_model = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Polynomial Regression"])

# Main Area
st.title("Student Academic Performance Analysis")

# Data loading
@st.cache_data
def load_data():
    # Mock data for demo purposes
    data = {
        "hours studied": [5, 8, 2, 7, 4],
        "attendance": [90, 95, 70, 85, 80],
        "sleep hours": [7, 6, 5, 8, 6],
        "extracurricular activities": ["None", "Sports", "Music", "None", "Art"],
        "department": ["Math", "Physics", "Chemistry", "Math", "Biology"],
        "participation score": [80, 85, 60, 75, 70],
        "total_score": [82, 91, 65, 88, 76]
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

if df is not None:
    st.subheader("Sample of Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Feature Importance Placeholder
    st.subheader("Feature Importance")
    st.info("Feature importance visualization will appear here after model training.")
    # TODO: Insert bar chart for feature importance
    
    # Model Performance Placeholder
    st.subheader("Model Performance")
    st.info("Model performance metrics will appear here after training.")
    # TODO: Insert accuracy, loss, etc.
    
    st.subheader("Try Custom Prediction")
    with st.form("predict_form"):
        input_dict = {}
        for feat in selected_features:
            if feat == "department":
                input_dict[feat] = st.selectbox("Department", sorted(df[feat].unique()))
            elif feat == "extracurricular activities":
                input_dict[feat] = st.selectbox("Extracurricular Activities", sorted(df[feat].unique()))
            else:
                input_dict[feat] = st.number_input(feat.title(), float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))
        submitted = st.form_submit_button("Predict Score")
        if submitted:
            st.warning("Prediction functionality coming soon.")
    
    st.subheader("Recommendations")
    st.info("Actionable insights based on feature importance will be shown here.")
else:
    st.stop()