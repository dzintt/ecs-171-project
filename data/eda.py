import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

PLOTS_DIR = "plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

numerical_features = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
    'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score',
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income',
    'Teacher_Quality', 'School_Type', 'Peer_Influence',
    'Learning_Disabilities', 'Parental_Education_Level', 
    'Distance_from_Home', 'Gender'
]

def detect_outliers(dataset: pd.DataFrame) -> pd.DataFrame:
    available_features = [f for f in numerical_features if f in dataset.columns]
    numerical_data = dataset[available_features].copy()
    Q1 = numerical_data.quantile(0.25)
    Q2 = numerical_data.quantile(0.5)
    Q3 = numerical_data.quantile(0.75)
    IQR = Q3 - Q1
    min_vals = numerical_data.min()
    max_vals = numerical_data.max()

    n_features = len(available_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(16, 4 * n_rows))
    for i, feature in enumerate(available_features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(x=numerical_data[feature])
        plt.title(feature)

    plt.tight_layout()
    boxplot_path = os.path.join(PLOTS_DIR, "outliers_boxplots.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()

    outliers = (numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))
    print("\nOutliers per feature:\n", outliers.sum())
    mask = ~((numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))).any(axis=1)
    cleaned_dataset = dataset[mask]
    
    original_rows = len(dataset)
    final_rows = len(cleaned_dataset)
    removed_rows = original_rows - final_rows
    
    print(f"\nOUTLIER REMOVAL SUMMARY")
    print(f"  Original rows: {original_rows}")
    print(f"  Rows with outliers removed: {removed_rows}")
    print(f"  Final rows: {final_rows}")
    print(f"  Data retention: {(final_rows/original_rows)*100:.2f}%")
    
    return cleaned_dataset

def visualize_features(dataset: pd.DataFrame):
    available_features = [f for f in numerical_features if f in dataset.columns]
    n_features = len(available_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(16, 4 * n_rows))
    for i, feature in enumerate(available_features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(dataset[feature], kde=True)
        plt.title(feature)

    plt.tight_layout()
    histogram_path = os.path.join(PLOTS_DIR, "feature_histograms.png")
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature histograms saved to: {histogram_path}")

    numerical_data = dataset[available_features]
    print("\nFEATURE STATISTICS")
    print("=" * 50)
    print("\nMean:\n{} \n \nMedian:\n{} \n \nSTD:\n{}".format(numerical_data.mean(), numerical_data.median(), numerical_data.std()))

def visualize_features_z_score(dataset: pd.DataFrame):
    available_features = [f for f in numerical_features if f in dataset.columns]
    numerical_data = dataset[available_features]
    
    data_mean = numerical_data.mean()
    data_std = numerical_data.std()
    normalized_zscore = (numerical_data - data_mean) / data_std

    n_features = len(available_features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(16, 4 * n_rows))
    for i, feature in enumerate(available_features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(normalized_zscore[feature], kde=True)
        plt.title(feature)

    plt.tight_layout()
    zscore_path = os.path.join(PLOTS_DIR, "feature_histograms_zscore.png")
    plt.savefig(zscore_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Z-score normalized histograms saved to: {zscore_path}")

    print("\nZ-SCORE NORMALIZED STATISTICS")
    print("=" * 50)
    print("\nMean(z-score):\n{} \n \nMedian(z-score):\n{} \n \nSTD(z-score):\n{}".format(normalized_zscore.mean(), normalized_zscore.median(), normalized_zscore.std()))

def visualize_correlations(dataset: pd.DataFrame):
    available_features = [f for f in numerical_features if f in dataset.columns]
    numerical_data = dataset[available_features]
    
    print("Creating pairplot (this may take a moment with 20 features)...")
    
    pairplot = sns.pairplot(numerical_data)
    pairplot.fig.suptitle('Pair Plot of Features', y=1.02)
    pairplot_path = os.path.join(PLOTS_DIR, "feature_pairplot.png")
    pairplot.savefig(pairplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pair plot saved to: {pairplot_path}")

    correlation_matrix = numerical_data.corr(method='pearson')
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    heatmap_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_path}")

def show_saved_plots():
    print("\nSAVED PLOTS SUMMARY")
    print("=" * 50)
    
    plot_files = [
        ("outliers_boxplots.png", "Box plots for outlier detection"),
        ("feature_histograms.png", "Feature distribution histograms"),
        ("feature_histograms_zscore.png", "Z-score normalized histograms"),
        ("feature_pairplot.png", "Pairwise feature relationships"),
        ("correlation_heatmap.png", "Feature correlation matrix")
    ]
    
    for filename, description in plot_files:
        filepath = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filename} ({file_size:.1f} MB)")
            print(f"   {description}")
        else:
            print(f"✗ {filename} (not found)")
    
    print(f"\nAll plots saved in: {os.path.abspath(PLOTS_DIR)}")

def run_full_eda(dataset: pd.DataFrame) -> pd.DataFrame:
    print("\nSTARTING EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    print("\n1. Generating feature distribution plots...")
    visualize_features(dataset)
    
    print("\n2. Generating z-score normalized plots...")
    visualize_features_z_score(dataset)
    
    print("\n3. Detecting and removing outliers...")
    cleaned_dataset = detect_outliers(dataset)
    
    print("\n4. Generating correlation analysis...")
    visualize_correlations(cleaned_dataset)
    
    print("\n5. EDA Summary...")
    show_saved_plots()
    
    print("\nEXPLORATORY DATA ANALYSIS COMPLETED!")
    print("=" * 60)
    
    return cleaned_dataset
