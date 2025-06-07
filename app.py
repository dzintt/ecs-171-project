import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from PIL import Image
import io

from data.dataset import load_data, clean_null_values, filter_features
from model.score_predictor import ScorePredictionModel
from plots.model_comparison_charts import create_model_performance_charts
from plots.model_diagnostics import create_comprehensive_model_diagnostics, plot_residuals

st.set_page_config(
    page_title="Student Exam Score Prediction Model",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b8daff;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'charts_generated' not in st.session_state:
    st.session_state.charts_generated = False

def load_and_prepare_data():
    with st.spinner("Loading dataset..."):
        dataset = load_data()
        cleaned_dataset = clean_null_values(dataset)
        filtered_dataset = filter_features(cleaned_dataset, features_to_keep=[
            "Attendance", "Hours_Studied", "Sleep_Hours", "Tutoring_Sessions",
            "Access_to_Resources", "Extracurricular_Activities", "Teacher_Quality", "Exam_Score"
        ])
    return filtered_dataset

def train_model_ui(dataset, kfold_splits):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Initializing model...")
    progress_bar.progress(10)
    
    model = ScorePredictionModel(dataset, kfold_splits=kfold_splits)
    
    status_text.text("Training models...")
    progress_bar.progress(30)
    
    model.train()
    
    status_text.text("Training completed!")
    progress_bar.progress(100)
    
    return model

def display_model_metrics(model):
    if model and model.models_performance:
        st.markdown("### Model Performance Summary")

        performance_data = model.get_performance_data()
        df_performance = pd.DataFrame({
            'Model': performance_data['Model'],
            'CV R²': performance_data['CV_R2'],
            'Test R²': performance_data['Test_R2'],
            'Test RMSE': performance_data['Test_RMSE'],
            'Test MAE': performance_data['Test_MAE']
        })
        
        best_model_name = model._get_best_model_name()
        best_performance = model.models_performance[best_model_name]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Model", best_model_name)
        with col2:
            st.metric("CV R²", f"{best_performance['cv_r2_mean']:.4f}")
        with col3:
            st.metric("Test R²", f"{best_performance['test_r2']:.4f}")
        with col4:
            st.metric("RMSE", f"{best_performance['test_rmse']:.4f}")
        
        st.markdown("### Detailed Performance Comparison")
        st.dataframe(df_performance.round(4), use_container_width=True)
        
        return df_performance
    return None

def display_charts():
    st.markdown("### Performance Visualizations")
    
    chart_files = []
    plot_dir = "plots"
    
    if os.path.exists(plot_dir):
        chart_patterns = [
            "model_performance_comparison_*fold.png",
            "model_performance_summary_*fold.png",
            "actual_vs_predicted_*fold.png",
            "training_curves_*fold.png",
            "residual_analysis_*fold.png"
        ]
        
        for pattern in chart_patterns:
            chart_files.extend(glob.glob(os.path.join(plot_dir, pattern)))
    
    if chart_files:
        chart_files.sort(key=os.path.getmtime, reverse=True)

        chart_names = []
        chart_images = []
        
        for file_path in chart_files[:6]: 
            if os.path.exists(file_path):
                try:
                    img = Image.open(file_path)
                    chart_images.append(img)

                    filename = os.path.basename(file_path)
                    if "comparison" in filename:
                        chart_names.append("Model Comparison")
                    elif "summary" in filename:
                        chart_names.append("Performance Summary")
                    elif "actual_vs_predicted" in filename:
                        chart_names.append("Actual vs Predicted")
                    elif "training_curves" in filename:
                        chart_names.append("Training Curves")
                    elif "residual" in filename:
                        chart_names.append("Residual Analysis")
                    else:
                        chart_names.append("Chart")
                except Exception as e:
                    st.error(f"Error loading chart {file_path}: {e}")
        
        if chart_images:
            tabs = st.tabs(chart_names)
            for i, (tab, img) in enumerate(zip(tabs, chart_images)):
                with tab:
                    st.image(img, use_container_width=True)
        else:
            st.info("No charts found. Train a model first to generate visualizations.")
    else:
        st.info("No charts available. Train a model and generate charts to see visualizations here.")

def prediction_interface(model, dataset):
    st.markdown("### Make Predictions")
    
    if model is None:
        st.warning("Please train a model first to make predictions.")
        return
    
    with st.form("prediction_form"):
        st.markdown("#### Enter Student Information:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            attendance = st.slider("Attendance (%)", 0, 100, 85)
            hours_studied = st.slider("Hours Studied (per week)", 0, 50, 20)
            sleep_hours = st.slider("Sleep Hours (per night)", 0, 12, 7)
            tutoring_sessions = st.slider("Tutoring Sessions (per month)", 0, 20, 4)
        
        with col2:
            access_resources = st.selectbox("Access to Resources", [1, 2, 3], 
                                          format_func=lambda x: ["Low", "Medium", "High"][x-1])
            extracurricular = st.selectbox("Extracurricular Activities", [0, 1], 
                                         format_func=lambda x: ["No", "Yes"][x])
            teacher_quality = st.selectbox("Teacher Quality", [1, 2, 3], 
                                         format_func=lambda x: ["Low", "Medium", "High"][x-1])
        
        predict_button = st.form_submit_button("Predict Exam Score")
        
        if predict_button:
            input_data = np.array([[
                attendance, hours_studied, sleep_hours, tutoring_sessions,
                access_resources, extracurricular, teacher_quality
            ]])
            
            try:

                prediction = model.predict(input_data)[0]

                st.markdown("---")
                st.markdown("### Prediction Result")
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 2rem;
                    border-radius: 15px;
                    text-align: center;
                    margin: 1.5rem 0;
                    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                ">
                    <h1 style="
                        color: white;
                        font-size: 3.5rem;
                        margin: 0;
                        font-weight: bold;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                    ">{prediction:.1f}</h1>
                    <h2 style="
                        color: #f0f0f0;
                        font-size: 1.3rem;
                        margin: 0.5rem 0 0 0;
                        font-weight: 300;
                    ">Predicted Exam Score</h2>
                </div>
                """, unsafe_allow_html=True)
        
            except Exception as e:
                st.error(f"Error making prediction: {e}")

def dataset_overview(dataset):
    st.markdown("### Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(dataset))
    with col2:
        st.metric("Features", len(dataset.columns) - 1)
    with col3:
        st.metric("Avg Score", f"{dataset['Exam_Score'].mean():.1f}")
    with col4:
        st.metric("Score Range", f"{dataset['Exam_Score'].min():.0f}-{dataset['Exam_Score'].max():.0f}")
    
    st.markdown("#### Sample Data")
    st.dataframe(dataset.head(10), use_container_width=True)
    
    st.markdown("#### Score Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.histogram(dataset, x='Exam_Score', nbins=30, 
                         title="Distribution of Exam Scores")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Summary Statistics:**")
        stats = dataset['Exam_Score'].describe().round(2)

        stat_col1, stat_col2 = st.columns(2)
        
        stat_items = list(stats.items())

        with stat_col1:
            for i in range(0, len(stat_items), 2):
                stat, value = stat_items[i]
                st.metric(stat.title(), f"{value:.1f}")
        
        with stat_col2:
            for i in range(1, len(stat_items), 2):
                if i < len(stat_items):
                    stat, value = stat_items[i]
                    st.metric(stat.title(), f"{value:.1f}")

def main():
    st.markdown('<h1 class="main-header">Student Exam Score Prediction Model</h1>', unsafe_allow_html=True)
    st.markdown("### ECS 171 Project - Team 23")
    st.markdown("By Justin Lin, Anson Tan, Jason Zhong, Raymond Wu, and Simon Yoo")
    
    with st.sidebar:
        st.title("Control Panel")

        if st.button("Load Dataset"):
            st.session_state.dataset = load_and_prepare_data()
            st.success("Dataset loaded successfully!")
        
        if st.session_state.dataset is not None:
            st.markdown("---")
            st.markdown("### Model Configuration")
            
            kfold_splits = st.selectbox(
                "Cross-Validation Folds",
                [5, 10, 20, 30],
                index=0,
                help="Higher values give more robust results but take longer to train"
            )
            
            if st.button("Train Model"):
                st.session_state.model = train_model_ui(st.session_state.dataset, kfold_splits)
                st.session_state.trained = True
                st.success("Model training completed!")
            
            if st.session_state.trained and st.button("Generate Charts"):
                with st.spinner("Generating visualizations..."):
                    try:
                        performance_data = st.session_state.model.get_performance_data()
                        create_model_performance_charts(performance_data)
                        
                        X = st.session_state.model.dataset["X"]
                        y = st.session_state.model.dataset["y"]
                        best_model = st.session_state.model.best_model
                        scaler = st.session_state.model.scaler
                        best_model_name = st.session_state.model._get_best_model_name()
                        
                        from sklearn.model_selection import KFold
                        cv_folds = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
                        
                        create_comprehensive_model_diagnostics(
                            best_model, X, y, scaler, best_model_name, kfold_splits, cv_folds
                        )
                        plot_residuals(
                            best_model, X, y, scaler, best_model_name, kfold_splits, cv_folds
                        )
                        
                        st.session_state.charts_generated = True
                        st.success("Charts generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating charts: {e}")
        

        if st.session_state.trained:
            st.markdown("---")
            st.markdown("### Model Info")
            best_model_name = st.session_state.model._get_best_model_name()
            st.info(f"**Best Model:** {best_model_name}")
            st.info(f"**CV Folds:** {st.session_state.model.kfold_splits}")
    
    if st.session_state.dataset is None:
        st.info("Click 'Load Dataset' in the sidebar to get started.")
        
        st.markdown("### About This Project")
        st.markdown("""
        Students are important in society as they will lead our future, solve problems and improve lives. As a result, education is important as it will shape their desired roles and help make them change the world. Student performance is used as a measure to evaluate and keep students on track to becoming bright contributors to society. In this project, we identify which factors most strongly impact academic outcomes so that we can help students improve their learning strategies in order to perform better academically.
        
        #### Dataset & Features
        Our analysis uses a comprehensive dataset of **6,378 student records** with **7 key academic and lifestyle factors**:
        
        **Key Factors Analyzed:**
        - **Attendance (%)** - Class attendance percentage
        - **Hours Studied** - Weekly study time commitment
        - **Sleep Hours** - Daily sleep duration for cognitive health
        - **Tutoring Sessions** - Additional academic support frequency
        - **Access to Resources** - Learning material and technology availability
        - **Extracurricular Activities** - Participation in non-academic activities
        - **Teacher Quality** - Instructor effectiveness rating
        
        The target variable is **Exam Score (0-100)**, allowing us to predict academic performance based on these behavioral and environmental factors.
                    
        #### Our Machine Learning Approach
        
        **Multiple Model Comparison:** We implement and compare 6 different regression algorithms to find the best predictor:
        - **Linear Regression** - Baseline linear relationship model
        - **Ridge Regression** - L2 regularization to prevent overfitting
        - **Lasso Regression** - L1 regularization with feature selection
        - **Random Forest** - Ensemble method using multiple decision trees
        - **Gradient Boosting** - Sequential boosting for improved accuracy
        - **K-Nearest Neighbors (KNN)** - Instance-based learning approach
        
        **Validation Process:**
        - **Cross-Validation:** Configurable K-fold validation (5, 10, 20, or 30 folds) for reliable performance estimation
        - **Hyperparameter Tuning:** GridSearchCV optimization for each algorithm
        - **Multiple Metrics:** Evaluation using R², RMSE, and MAE for comprehensive assessment
        - **Train-Test Split:** 80/20 split with stratified sampling to ensure representative data distribution
        
        **Advanced Analytics:**
        - **Model Diagnostics:** Training curves, actual vs predicted plots, and residual analysis
        - **Performance Visualization:** Comprehensive charts comparing all models across multiple metrics
        - **Feature Scaling:** StandardScaler normalization for optimal model performance
        - **Interactive Predictions:** Real-time score prediction with confidence intervals
        """)
        
    else:
        dataset_overview(st.session_state.dataset)
        
        if st.session_state.trained:
            display_model_metrics(st.session_state.model)
            prediction_interface(st.session_state.model, st.session_state.dataset)
            if st.session_state.charts_generated:
                display_charts()
        else:
            st.info("Configure and train your model using the sidebar controls.")

if __name__ == "__main__":
    main()