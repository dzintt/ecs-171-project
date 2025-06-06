import streamlit as st
import pandas as pd

st.set_page_config(page_title="Student Academic Performance Analysis", layout="wide")

# Sidebar
st.sidebar.title("Student Performance Analyzer")
st.sidebar.markdown("""
**Project Description**

Identify which factors most significantly influence student academic performance using ML models and a public dataset.

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
    
    # Feature Importance Visualization (Mock Data)
    st.subheader("Feature Importance")
    mock_importance = {
        "hours studied": 0.35,
        "attendance": 0.25,
        "sleep hours": 0.10,
        "extracurricular activities": 0.08,
        "department": 0.07,
        "participation score": 0.15
    }
    st.bar_chart(pd.DataFrame.from_dict(mock_importance, orient="index", columns=["Importance"]))
    
    # Model Performance (Mock Data)
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", "0.82")
    col2.metric("MAE", "3.5")
    col3.metric("RMSE", "4.2")

    # Mock scatterplot with line of best fit using Streamlit ECharts
    from streamlit_echarts import st_echarts
    import numpy as np

    # Mock true vs predicted data
    y_true = [82, 91, 65, 88, 76, 85, 78, 92, 73, 88, 95, 67, 80, 83, 90, 72, 86, 89, 81, 77]
    y_pred = [80, 89, 67, 85, 78, 83, 76, 90, 75, 86, 94, 70, 82, 85, 88, 74, 85, 87, 79, 78]
    scatter_data = list(zip(y_true, y_pred))

    # Calculate line of best fit
    m, b = np.polyfit(y_true, y_pred, 1)
    line_x = [min(y_true), max(y_true)]
    line_y = [m * x + b for x in line_x]

    min_val = min(min(y_true), min(y_pred)) - 2
    max_val = max(max(y_true), max(y_pred)) + 2
    options = {
        "xAxis": {
            "name": "Actual Total Score",
            "min": min_val,
            "max": max_val,
        },
        "yAxis": {
            "name": "Predicted Total Score",
            "min": min_val,
            "max": max_val,
        },
        "series": [
            {
                "symbolSize": 16,
                "data": scatter_data,
                "type": "scatter",
                "name": "Predictions",
                "itemStyle": {"color": "#1976D2"}
            },
            {
                "data": [[line_x[0], line_y[0]], [line_x[1], line_y[1]]],
                "type": "line",
                "name": "Best Fit",
                "lineStyle": {"color": "#E53935", "width": 3, "type": "solid"},
                "showSymbol": False,
            },
            {
                "data": [[min(y_true), min(y_true)], [max(y_true), max(y_true)]],
                "type": "line",
                "name": "Ideal",
                "lineStyle": {"color": "#888", "width": 2, "type": "dashed"},
                "showSymbol": False,
            }
        ],
        "legend": {"data": ["Predictions", "Best Fit", "Ideal"]},
        "tooltip": {"trigger": "axis"},
        "title": {"text": "Predicted vs Actual Scores"},
        "grid": {"left": "18%", "right": "18%", "top": 50, "bottom": 60},
    }
    st_echarts(options=options, height="400px")

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