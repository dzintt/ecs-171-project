import kagglehub
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("mahmoudelhemaly/students-grading-dataset")
files = os.listdir(path)
csv_files = [f for f in files if f.endswith('.csv')]
numerical_features = ['Study_Hours_per_Week', 'Attendance (%)', "Sleep_Hours_per_Night", "Participation_Score", "Final_Score"]

def load_data() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    return filter_features(df)

def filter_features(dataset: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        'Study_Hours_per_Week',           # hours studied
        'Attendance (%)',                 # attendance
        'Sleep_Hours_per_Night',          # sleep hours
        'Extracurricular_Activities',     # extracurricular activities
        'Department',                     # department
        'Participation_Score',            # participation score
        'Final_Score'                     # final score
    ]
    return dataset[[col for col in selected_columns if col in dataset.columns]]

# Just to check if we need to populate null entries or delete duplicate ones 
# (Pretty useless for our dataset since we have no duplicates or null values, but I thought I should document that we checked somehow)
def check_dupe_or_null(dataset: pd.DataFrame) -> bool:
    result = False
    # Makes a dataset containing the number of null values for each feature
    dataset = dataset.isnull().sum()
    # If any feature have more than 0 null values, return True
    for i, j in dataset.items():
        if j > 0:
            result = True
            print("Null values detected.")
            return result
    # If dataset had duplicates to drop, return True
    if not dataset.drop_duplicates(inplace=True) == None:
        result = True
        print("Duplicate entires detected.")
        return result
    
    # If no duplicates or null values are found, return false
    print("There exist no duplicate entries or null values.")
    return result

def remove_outliers(dataset: pd.DataFrame) -> pd.DataFrame:
    # First drop non numerical features
    dataset = dataset.drop(['Extracurricular_Activities', 'Department'], axis=1)

    # Using IQR method
    Q1 = dataset.quantile(0.25)
    Q2 = dataset.quantile(0.5)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    min = dataset.min()
    max = dataset.max()

    # Plot each feature as a box plot
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x=dataset[feature])
        plt.title(feature)

    print("\nQ1:\n{} \n \nQ2:\n{} \n \nQ3:\n{} \n \nIQR:\n{} \n \nMin:\n{} \n \nMax:\n{}".format(Q1, Q2, Q3, IQR, min, max))

    # Show how many outliers exist
    outliers = (dataset < (Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))
    print("\nOutliers:\n", outliers.sum())

    # Remove any outliers using IQR method
    dataset = dataset[~((dataset < (Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
    return dataset

# Plot each feature as histogram (can see skewedness and distribution)
def visualize_features(dataset: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(dataset[feature], kde=True)
        plt.title(feature)

    plt.tight_layout()
    plt.show()

    print("\nMean:\n{} \n \nMedian:\n{} \n \nSTD:\n{}".format(dataset[numerical_features].mean() , dataset[numerical_features].median(), dataset[numerical_features].std()))

# Normalize the data using z-score and plot as histogram for each feature
def visualize_features_z_score(dataset: pd.DataFrame):
    dataset = dataset.drop(['Extracurricular_Activities', 'Department'], axis=1)
    # Normalize using z-score
    data_mean = dataset.mean()
    data_std = dataset.std()
    normalized_zscore = (dataset - data_mean) / data_std

    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(2, 3, i)
        sns.histplot(normalized_zscore[feature], kde=True)
        plt.title(feature)

    plt.tight_layout()
    plt.show()

    print("\nMean(z-score):\n{} \n \nMedian(z-score):\n{} \n \nSTD(z-score):\n{}".format(normalized_zscore[numerical_features].mean() , normalized_zscore[numerical_features].median(), normalized_zscore[numerical_features].std()))

# Attempted correlation calculation with raw values, normalized using z-score, and normalized using min max
def visualize_corr(dataset: pd.DataFrame):
    # Normalize using z-score (optional, uncomment to try)*
    # data_mean = dataset.mean()
    # data_std = dataset.std()
    # normalized_zscore = (dataset - data_mean) / data_std

    # Normlize using min max (optional, uncomment to try)^
    # min = dataset.min()
    # max = dataset.max()
    # scaled = (dataset - min) / (max - min)

    # Show scatter plot for each feature with each other feature
    sns.pairplot(dataset)
    #* sns.pairplot(normalized_zscore)
    #^ sns.pairplot(scaled)
    plt.title('Pair Plot of Features')
    plt.show()

    correlation_matrix = dataset.corr(method='pearson')
    #* correlation_matrix = normalized_zscore.corr()
    #^ correlation_matrix = scaled.corr()

    # Visualize correlation matrix using heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix Heatmap')
    plt.show()


        