import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create plots directory if it doesn't exist
PLOTS_DIR = "plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
    print(f"📁 Created plots directory: {PLOTS_DIR}")

# All features after encoding (categorical values replaced with numerical ones)
numerical_features = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
    'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score',
    # Originally categorical features (now numerical after encoding)
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income',
    'Teacher_Quality', 'School_Type', 'Peer_Influence',
    'Learning_Disabilities', 'Parental_Education_Level', 
    'Distance_from_Home', 'Gender'
]

def detect_outliers(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and visualize outliers using box plots and IQR method.
    Returns the dataset with outliers removed.
    """
    # Keep only the numerical features (including encoded categorical features)
    numerical_data = dataset[numerical_features].copy()
    # Remove any columns that don't exist in the dataset
    numerical_data = numerical_data[[col for col in numerical_data.columns if col in dataset.columns]]

    # Using IQR method
    Q1 = numerical_data.quantile(0.25)
    Q2 = numerical_data.quantile(0.5)
    Q3 = numerical_data.quantile(0.75)
    IQR = Q3 - Q1
    min_vals = numerical_data.min()
    max_vals = numerical_data.max()

    # Plot each feature as a box plot
    available_features = [f for f in numerical_features if f in numerical_data.columns]
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

    # Show how many outliers exist
    outliers = (numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))
    print("\nOutliers per feature:\n", outliers.sum())

    # Remove any outliers using IQR method
    mask = ~((numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))).any(axis=1)
    cleaned_dataset = dataset[mask]
    
    original_rows = len(dataset)
    final_rows = len(cleaned_dataset)
    removed_rows = original_rows - final_rows
    
    print(f"\n🧹 OUTLIER REMOVAL SUMMARY")
    print(f"  • Original rows: {original_rows}")
    print(f"  • Rows with outliers removed: {removed_rows}")
    print(f"  • Final rows: {final_rows}")
    print(f"  • Data retention: {(final_rows/original_rows)*100:.2f}%")
    
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
    print(f"📊 Feature histograms saved to: {histogram_path}")

    numerical_data = dataset[available_features]
    print("\n📊 FEATURE STATISTICS")
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
    print(f"📊 Z-score normalized histograms saved to: {zscore_path}")

    print("\n📊 Z-SCORE NORMALIZED STATISTICS")
    print("=" * 50)
    print("\nMean(z-score):\n{} \n \nMedian(z-score):\n{} \n \nSTD(z-score):\n{}".format(normalized_zscore.mean(), normalized_zscore.median(), normalized_zscore.std()))

def visualize_correlations(dataset: pd.DataFrame):
    """
    Create correlation analysis with pairplot and heatmap.
    """
    # Keep only numerical features for correlation analysis (including encoded categorical features)
    available_features = [f for f in numerical_features if f in dataset.columns]
    numerical_data = dataset[available_features]
    
    # Normalize using z-score (optional, uncomment to try)*
    # data_mean = numerical_data.mean()
    # data_std = numerical_data.std()
    # normalized_zscore = (numerical_data - data_mean) / data_std

    # Normlize using min max (optional, uncomment to try)^
    # min_vals = numerical_data.min()
    # max_vals = numerical_data.max()
    # scaled = (numerical_data - min_vals) / (max_vals - min_vals)

    print("🔄 Creating pairplot (this may take a moment with 20 features)...")
    
    # Show scatter plot for each feature with each other feature
    pairplot = sns.pairplot(numerical_data)
    #* sns.pairplot(normalized_zscore)
    #^ sns.pairplot(scaled)
    pairplot.fig.suptitle('Pair Plot of Features', y=1.02)
    pairplot_path = os.path.join(PLOTS_DIR, "feature_pairplot.png")
    pairplot.savefig(pairplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Pair plot saved to: {pairplot_path}")

    correlation_matrix = numerical_data.corr(method='pearson')
    #* correlation_matrix = normalized_zscore.corr()
    #^ correlation_matrix = scaled.corr()

    # Visualize correlation matrix using heatmap
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
    print(f"📊 Correlation heatmap saved to: {heatmap_path}")

def show_saved_plots():
    """
    Display information about all saved plots.
    """
    print("\n📁 SAVED PLOTS SUMMARY")
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
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"✅ {filename} ({file_size:.1f} MB)")
            print(f"   📝 {description}")
        else:
            print(f"❌ {filename} (not found)")
    
    print(f"\n📂 All plots saved in: {os.path.abspath(PLOTS_DIR)}")

def run_full_eda(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Run complete exploratory data analysis pipeline.
    Returns cleaned dataset with outliers removed.
    """
    print("\n🔍 STARTING EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # 1. Feature distributions
    print("\n1️⃣ Generating feature distribution plots...")
    visualize_features(dataset)
    
    # 2. Z-score normalized distributions
    print("\n2️⃣ Generating z-score normalized plots...")
    visualize_features_z_score(dataset)
    
    # 3. Outlier detection and removal
    print("\n3️⃣ Detecting and removing outliers...")
    cleaned_dataset = detect_outliers(dataset)
    
    # 4. Correlation analysis
    print("\n4️⃣ Generating correlation analysis...")
    visualize_correlations(cleaned_dataset)
    
    # 5. Summary
    print("\n5️⃣ EDA Summary...")
    show_saved_plots()
    
    print("\n🎉 EXPLORATORY DATA ANALYSIS COMPLETED!")
    print("=" * 60)
    
    return cleaned_dataset
