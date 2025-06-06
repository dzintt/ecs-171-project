import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd
import numpy as np

path = kagglehub.dataset_download("lainguyn123/student-performance-factors")
files = os.listdir(path)
csv_files = [f for f in files if f.endswith('.csv')]

def encode_categorical_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical variables to numerical values by directly replacing the original values.
    """
    df = dataset.copy()
    
    # Binary encodings (Yes/No -> 1/0)
    binary_mappings = {
        'Extracurricular_Activities': {'Yes': 1, 'No': 0},
        'Internet_Access': {'Yes': 1, 'No': 0},
        'Learning_Disabilities': {'Yes': 1, 'No': 0}
    }
    
    for column, mapping in binary_mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)
    
    # Ordinal encodings (Low/Medium/High -> 1/2/3)
    ordinal_mappings = {
        'Parental_Involvement': {'Low': 1, 'Medium': 2, 'High': 3},
        'Access_to_Resources': {'Low': 1, 'Medium': 2, 'High': 3},
        'Motivation_Level': {'Low': 1, 'Medium': 2, 'High': 3},
        'Family_Income': {'Low': 1, 'Medium': 2, 'High': 3},
        'Teacher_Quality': {'Low': 1, 'Medium': 2, 'High': 3}
    }
    
    for column, mapping in ordinal_mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)
    
    # Custom ordinal encodings
    custom_mappings = {
        'Parental_Education_Level': {'High School': 1, 'College': 2, 'Postgraduate': 3},
        'Distance_from_Home': {'Near': 1, 'Moderate': 2, 'Far': 3},
        'Peer_Influence': {'Negative': 1, 'Neutral': 2, 'Positive': 3}
    }
    
    for column, mapping in custom_mappings.items():
        if column in df.columns:
            # Handle missing values by checking available keys
            available_values = df[column].dropna().unique()
            filtered_mapping = {k: v for k, v in mapping.items() if k in available_values}
            df[column] = df[column].map(filtered_mapping)
    
    # Binary categorical encodings
    if 'School_Type' in df.columns:
        df['School_Type'] = df['School_Type'].map({'Public': 0, 'Private': 1})
    
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    # Count encoded features
    categorical_features = [
        'Extracurricular_Activities', 'Parental_Involvement', 'Access_to_Resources',
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    encoded_count = len([col for col in categorical_features if col in df.columns])

    return df

def show_dataset_summary(dataset: pd.DataFrame) -> None:
    """
    Display a comprehensive summary of the dataset structure.
    """
    print("\n📋 DATASET SUMMARY")
    print("=" * 50)
    print(f"📊 Total rows: {len(dataset)}")
    print(f"📊 Total columns: {len(dataset.columns)}")
    
    # Categorize columns
    originally_numerical = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                           'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']
    originally_categorical = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                             'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                             'School_Type', 'Peer_Influence', 'Learning_Disabilities', 
                             'Parental_Education_Level', 'Distance_from_Home', 'Gender']
    
    numerical_present = [col for col in originally_numerical if col in dataset.columns]
    categorical_present = [col for col in originally_categorical if col in dataset.columns]
    
    print(f"\n📈 Originally numerical features: {len(numerical_present)}")
    for col in numerical_present:
        print(f"  • {col}")
    
    print(f"\n🔢 Originally categorical features (now numerical): {len(categorical_present)}")
    for col in categorical_present:
        # Show sample values to confirm encoding
        sample_values = sorted(dataset[col].dropna().unique())
        print(f"  • {col} (values: {sample_values})")
    
    print(f"\n✨ Total numerical features: {len(numerical_present) + len(categorical_present)}")

def load_data() -> pd.DataFrame:
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "lainguyn123/student-performance-factors",
        "StudentPerformanceFactors.csv",
    )
    df = encode_categorical_features(df)
    return df

def filter_features(dataset: pd.DataFrame, features_to_keep: list[str]) -> pd.DataFrame:
    filtered_columns = [col for col in features_to_keep if col in dataset.columns]
    return dataset[filtered_columns]

def clean_null_values(dataset: pd.DataFrame) -> pd.DataFrame:
    original_rows = len(dataset)
    null_counts = dataset.isnull().sum()
    total_nulls = null_counts.sum()
    
    if total_nulls == 0:
        return dataset

    for column, null_count in null_counts.items():
        if null_count > 0:
            percentage = (null_count / original_rows) * 100
    
    rows_with_nulls = dataset.isnull().any(axis=1)
    rows_to_remove = rows_with_nulls.sum()
    

    cleaned_dataset = dataset.dropna()
    final_rows = len(cleaned_dataset)
    
    # print(f"\n✨ Cleaning completed!")
    # print(f"  • Original rows: {original_rows}")
    # print(f"  • Rows removed: {rows_to_remove}")
    # print(f"  • Final rows: {final_rows}")
    # print(f"  • Data retention: {(final_rows/original_rows)*100:.2f}%")
    
    return cleaned_dataset

def remove_outliers_simple(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers using IQR method without visualization.
    For visualization, use eda.detect_outliers() instead.
    """
    # All features after encoding (categorical values replaced with numerical ones)
    numerical_features = [
        'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
        'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score',
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
        'Motivation_Level', 'Internet_Access', 'Family_Income',
        'Teacher_Quality', 'School_Type', 'Peer_Influence',
        'Learning_Disabilities', 'Parental_Education_Level', 
        'Distance_from_Home', 'Gender'
    ]
    
    available_features = [f for f in numerical_features if f in dataset.columns]
    numerical_data = dataset[available_features].copy()

    Q1 = numerical_data.quantile(0.25)
    Q3 = numerical_data.quantile(0.75)
    IQR = Q3 - Q1

    mask = ~((numerical_data < (Q1 - 1.5 * IQR)) | (numerical_data > (Q3 + 1.5 * IQR))).any(axis=1)
    cleaned_dataset = dataset[mask]
    
    original_rows = len(dataset)
    final_rows = len(cleaned_dataset)
    removed_rows = original_rows - final_rows

    
    return cleaned_dataset