import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("lainguyn123/student-performance-factors")
files = os.listdir(path)
csv_files = [f for f in files if f.endswith('.csv')]

numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']

def load_data() -> pd.DataFrame:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "lainguyn123/student-performance-factors",
        "StudentPerformanceFactors.csv",  # Common CSV filename pattern
    )
    return filter_features(df)

def filter_features(dataset: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        'Hours_Studied',                  # hours studied per week
        'Attendance',                     # attendance percentage
        'Sleep_Hours',                    # sleep hours per night
        'Extracurricular_Activities',     # extracurricular activities
        'Parental_Involvement',           # parental involvement level
        'Access_to_Resources',            # access to educational resources
        'Previous_Scores',                # previous exam scores
        'Motivation_Level',               # motivation level
        'Internet_Access',                # internet access
        'Tutoring_Sessions',              # tutoring sessions per month
        'Family_Income',                  # family income level
        'Teacher_Quality',                # teacher quality
        'School_Type',                    # school type
        'Physical_Activity',              # physical activity hours
        'Gender',                         # gender
        'Exam_Score'                      # final exam score
    ]
    return dataset[[col for col in selected_columns if col in dataset.columns]]

def check_dupe_or_null(dataset: pd.DataFrame) -> bool:
    result = False
    total_rows = len(dataset)
    
    print(f"Dataset Analysis - Total rows: {total_rows}")
    print("=" * 50)
    
    null_counts = dataset.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls > 0:
        result = True
        print(f"⚠️  NULL VALUES DETECTED - Total: {total_nulls}")
        print("\nNull values by column:")
        for column, null_count in null_counts.items():
            if null_count > 0:
                percentage = (null_count / total_rows) * 100
                print(f"  • {column}: {null_count} nulls ({percentage:.2f}% of data)")
        
        print(f"\nColumns with null values: {(null_counts > 0).sum()} out of {len(dataset.columns)}")
    else:
        print("✅ No null values found in any column")

    original_length = len(dataset)
    dataset_no_dupes = dataset.drop_duplicates()
    duplicate_count = original_length - len(dataset_no_dupes)
    
    if duplicate_count > 0:
        result = True
        percentage_dupes = (duplicate_count / original_length) * 100
        print(f"\n⚠️  DUPLICATE ENTRIES DETECTED")
        print(f"  • Number of duplicate rows: {duplicate_count}")
        print(f"  • Percentage of duplicates: {percentage_dupes:.2f}%")
        print(f"  • Unique rows: {len(dataset_no_dupes)}")
    else:
        print("\n✅ No duplicate entries found")
    
    # Summary
    print("\n" + "=" * 50)
    if not result:
        print("🎉 Dataset is clean - no null values or duplicates!")
    else:
        print("⚠️  Dataset requires cleaning - see details above")
    
    return result

def remove_outliers(dataset: pd.DataFrame) -> pd.DataFrame:
    # First drop non numerical features
    categorical_cols = ['Extracurricular_Activities', 'Parental_Involvement', 'Access_to_Resources', 
                       'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
                       'School_Type', 'Gender']
    numerical_data = dataset.drop([col for col in categorical_cols if col in dataset.columns], axis=1)

    # Using IQR method
    Q1 = numerical_data.quantile(0.25)
    Q2 = numerical_data.quantile(0.5)
    Q3 = numerical_data.quantile(0.75)
    IQR = Q3 - Q1
    min_vals = numerical_data.min()
    max_vals = numerical_data.max()

    # Plot each feature as a box plot
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(numerical_features, 1):
        if feature in numerical_data.columns:
            plt.subplot(2, 4, i)
            sns.boxplot(x=numerical_data[feature])
            plt.title(feature)

    print("\nQ1:\n{} \n \nQ2:\n{} \n \nQ3:\n{} \n \nIQR:\n{} \n \nMin:\n{} \n \nMax:\n{}".format(Q1, Q2, Q3, IQR, min_vals, max_vals))

    # Show how many outliers exist
    outliers = (numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))
    print("\nOutliers:\n", outliers.sum())

    # Remove any outliers using IQR method
    mask = ~((numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))).any(axis=1)
    return dataset[mask]

# Plot each feature as histogram (can see skewedness and distribution)
def visualize_features(dataset: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(numerical_features, 1):
        if feature in dataset.columns:
            plt.subplot(2, 4, i)
            sns.histplot(dataset[feature], kde=True)
            plt.title(feature)

    plt.tight_layout()
    plt.show()

    numerical_data = dataset[numerical_features]
    print("\nMean:\n{} \n \nMedian:\n{} \n \nSTD:\n{}".format(numerical_data.mean(), numerical_data.median(), numerical_data.std()))

# Normalize the data using z-score and plot as histogram for each feature
def visualize_features_z_score(dataset: pd.DataFrame):
    # Keep only numerical features
    categorical_cols = ['Extracurricular_Activities', 'Parental_Involvement', 'Access_to_Resources', 
                       'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
                       'School_Type', 'Gender']
    numerical_data = dataset.drop([col for col in categorical_cols if col in dataset.columns], axis=1)
    
    # Normalize using z-score
    data_mean = numerical_data.mean()
    data_std = numerical_data.std()
    normalized_zscore = (numerical_data - data_mean) / data_std

    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(numerical_features, 1):
        if feature in normalized_zscore.columns:
            plt.subplot(2, 4, i)
            sns.histplot(normalized_zscore[feature], kde=True)
            plt.title(feature)

    plt.tight_layout()
    plt.show()

    print("\nMean(z-score):\n{} \n \nMedian(z-score):\n{} \n \nSTD(z-score):\n{}".format(normalized_zscore.mean(), normalized_zscore.median(), normalized_zscore.std()))

# Attempted correlation calculation with raw values, normalized using z-score, and normalized using min max
def visualize_corr(dataset: pd.DataFrame):
    # Keep only numerical features for correlation analysis
    categorical_cols = ['Extracurricular_Activities', 'Parental_Involvement', 'Access_to_Resources', 
                       'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
                       'School_Type', 'Gender']
    numerical_data = dataset.drop([col for col in categorical_cols if col in dataset.columns], axis=1)
    
    # Normalize using z-score (optional, uncomment to try)*
    # data_mean = numerical_data.mean()
    # data_std = numerical_data.std()
    # normalized_zscore = (numerical_data - data_mean) / data_std

    # Normlize using min max (optional, uncomment to try)^
    # min_vals = numerical_data.min()
    # max_vals = numerical_data.max()
    # scaled = (numerical_data - min_vals) / (max_vals - min_vals)

    # Show scatter plot for each feature with each other feature
    sns.pairplot(numerical_data)
    #* sns.pairplot(normalized_zscore)
    #^ sns.pairplot(scaled)
    plt.suptitle('Pair Plot of Features', y=1.02)
    plt.show()

    correlation_matrix = numerical_data.corr(method='pearson')
    #* correlation_matrix = normalized_zscore.corr()
    #^ correlation_matrix = scaled.corr()

    # Visualize correlation matrix using heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    plt.show()


        