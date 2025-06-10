# Student Exam Score Prediction Model

**ECS 171 Project - Team 23**

## 🎯 Project Overview

This project implements a comprehensive machine learning system to predict student exam scores based on academic and lifestyle factors. Our goal is to identify which factors most strongly impact academic outcomes to help students improve their learning strategies and perform better academically.

### Key Features

-   **6 ML Algorithm Comparison**: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, KNN
-   **Robust Cross-Validation**: Configurable K-fold validation (5, 10, 20, 30 folds)
-   **Hyperparameter Optimization**: GridSearchCV for all algorithms
-   **Interactive Web Interface**: Streamlit-based GUI for easy interaction
-   **Comprehensive Visualizations**: Model performance charts, diagnostic plots, and residual analysis
-   **Command Line Interface**: Alternative CLI for batch processing

## 📊 Dataset

Our analysis uses a comprehensive dataset of **6,378 student records** with **7 key features**:

| Feature                        | Description                                    | Type        |
| ------------------------------ | ---------------------------------------------- | ----------- |
| **Attendance**                 | Class attendance percentage (0-100%)           | Continuous  |
| **Hours_Studied**              | Weekly study time commitment                   | Continuous  |
| **Sleep_Hours**                | Daily sleep duration for cognitive health      | Continuous  |
| **Tutoring_Sessions**          | Additional academic support frequency          | Continuous  |
| **Access_to_Resources**        | Learning material availability (1-3)           | Categorical |
| **Extracurricular_Activities** | Participation in non-academic activities (0-1) | Binary      |
| **Teacher_Quality**            | Instructor effectiveness rating (1-3)          | Categorical |

**Target Variable**: Exam Score (0-100)

## 🚀 Quick Start

### Prerequisites

-   Python 3.7 or higher
-   pip package manager

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd ecs-171-project
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Verify installation**

```bash
python -c "import streamlit, sklearn, pandas; print('All dependencies installed successfully!')"
```

## 💻 Usage

### Option 1: Web Interface (Recommended)

Launch the interactive Streamlit web application:

```bash
streamlit run app.py
```

This will open your browser to `http://localhost:8501` with a user-friendly interface featuring:

-   **Dataset Loading & Overview**: Load and explore the student performance dataset
-   **Model Training**: Configure cross-validation and train multiple ML models
-   **Performance Analysis**: View comprehensive model comparison charts
-   **Interactive Predictions**: Make real-time predictions with custom student parameters
-   **Visualization Gallery**: Browse generated charts and diagnostic plots

### Option 2: Command Line Interface

For batch processing or automated workflows:

```bash
python main.py
```

Interactive menu options:

1. **Load Dataset**: Load and preprocess the student data
2. **Train Models**: Train all 6 algorithms with cross-validation
3. **Generate Charts**: Create performance comparison visualizations
4. **Model Diagnostics**: Generate detailed diagnostic plots
5. **Make Predictions**: Predict scores for custom inputs

## 📁 Project Structure

```
ecs-171-project/
├── app.py                              # Streamlit web application
├── main.py                             # Command line interface
├── requirements.txt                    # Python dependencies
├── StudentPerformanceFactors.csv       # Dataset file
├── README.md                           # This file
│
├── data/                               # Data processing modules
│   ├── dataset.py                      # Data loading and preprocessing
│   ├── eda.py                          # Exploratory data analysis
│   └── report_charts.py                # Dataset analysis charts
│
├── model/                              # Machine learning modules
│   └── score_predictor.py              # ML model implementation
│
├── plots/                              # Visualization modules & outputs
│   ├── model_comparison_charts.py      # Model performance charts
│   ├── model_diagnostics.py            # Diagnostic visualizations
│   └── *.png                          # Generated chart files
│
├── saved_models/                       # Trained model storage
│   └── *.pkl                          # Pickled model files
│
└── utils/                              # Utility functions
```

## 🔬 Technical Implementation

### Machine Learning Pipeline

1. **Data Preprocessing**

    - Missing value handling and outlier detection
    - Feature scaling using StandardScaler
    - Train-test split (80/20) with stratified sampling

2. **Model Training & Comparison**

    - **Linear Regression**: Baseline linear relationship model
    - **Ridge Regression**: L2 regularization to prevent overfitting
    - **Lasso Regression**: L1 regularization with feature selection
    - **Random Forest**: Ensemble method using multiple decision trees
    - **Gradient Boosting**: Sequential boosting for improved accuracy
    - **K-Nearest Neighbors**: Instance-based learning approach

3. **Validation & Optimization**

    - K-fold cross-validation (configurable: 5, 10, 20, 30 folds)
    - GridSearchCV hyperparameter tuning for each algorithm
    - Multiple evaluation metrics: R², RMSE, MAE

4. **Model Selection**
    - Automatic best model selection based on cross-validation R² score
    - Performance comparison across all algorithms
    - Confidence interval estimation for predictions

### Visualization Suite

-   **Model Performance Charts**: Side-by-side comparison of all algorithms
-   **Training Curves**: Learning progression for tree-based models
-   **Actual vs Predicted**: Scatter plots with regression lines
-   **Residual Analysis**: 4-plot diagnostic grid for model validation
-   **Feature Distributions**: Histograms and statistical summaries

## 📈 Results & Performance

The system automatically identifies the best-performing model based on cross-validation scores. Typical performance metrics include:

-   **R² Score**: 0.85-0.95 (depending on algorithm and data split)
-   **RMSE**: 3-7 points (on 0-100 scale)
-   **MAE**: 2-5 points average error

_Note: Actual performance may vary based on data preprocessing and hyperparameter settings._

## 🔧 Customization

### Modifying Cross-Validation

Edit the K-fold options in `app.py` or `main.py`:

```python
kfold_splits = st.selectbox("Cross-Validation Folds", [5, 10, 20, 30])
```

### Adding New Algorithms

Extend the model dictionary in `model/score_predictor.py`:

```python
self.models = {
    'Your_New_Model': YourAlgorithm(),
    # ... existing models
}
```

### Custom Hyperparameters

Modify the parameter grids in `model/score_predictor.py`:

```python
param_grids = {
    'Your_Model': {
        'parameter_name': [value1, value2, value3]
    }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **ImportError: Module not found**

    ```bash
    pip install -r requirements.txt
    ```

2. **Streamlit port already in use**

    ```bash
    streamlit run app.py --server.port 8502
    ```

3. **Memory issues with large cross-validation**

    - Reduce K-fold splits to 5 or 10
    - Use smaller parameter grids for GridSearchCV

4. **Charts not displaying**
    - Ensure `plots/` directory exists
    - Check file permissions for writing charts

## 📚 Dependencies

Core packages and their purposes:

-   **streamlit**: Web application framework
-   **scikit-learn**: Machine learning algorithms and tools
-   **pandas**: Data manipulation and analysis
-   **matplotlib/seaborn**: Data visualization
-   **plotly**: Interactive charts
-   **numpy**: Numerical computing
-   **scipy**: Scientific computing utilities

## 🤝 Contributing

This is an academic project for ECS 171. For educational use and reference only.

## 📄 License

This project is created for educational purposes as part of UC Davis ECS 171 coursework.

---

**Team 23 - UC Davis ECS 171**
