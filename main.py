from data.dataset import load_data, clean_null_values, filter_features, show_dataset_summary
from data.eda import run_full_eda
from model.train import train_model


def load_and_clean_data():
    dataset = load_data()
    cleaned_dataset = clean_null_values(dataset)
    cleaned_dataset = filter_features(cleaned_dataset, features_to_keep=[
        "Attendance",
        "Hours_Studied",
        "Sleep_Hours",
        "Tutoring_Sessions",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Teacher_Quality",
    ])
    return cleaned_dataset


def main():
    dataset = load_and_clean_data()
    print("What would you like to do?")
    print("1. Run EDA")
    print("2. Train Model")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        run_full_eda(dataset)
    elif choice == "2":
        train_model(dataset)
    else:
        print("Invalid selection. Please enter 1 or 2.")

if __name__ == "__main__":
    main()


# %%
