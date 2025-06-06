from data.dataset import load_data, show_dataset_summary, clean_null_values
from data.eda import run_full_eda
from model.train import train_model
from pick import pick


def load_and_clean_data():
    dataset = load_data()
    cleaned_dataset = clean_null_values(dataset)
    return cleaned_dataset


def main():
    dataset = load_and_clean_data()
    selection = pick(["Run EDA", "Train Model"], "What would you like to do?")
    if selection[0] == "Run EDA":
        run_full_eda(dataset)
    elif selection[0] == "Train Model":
        train_model(dataset)

if __name__ == "__main__":
    main()


# %%
