import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download("mahmoudelhemaly/students-grading-dataset")
files = os.listdir(path)
csv_files = [f for f in files if f.endswith('.csv')]

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