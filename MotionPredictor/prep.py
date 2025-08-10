import pandas as pd
from collect import DATA_ROOT
import os

def split_csv_train_validation(file_path, validation_split: float, seed=42):
    df = pd.read_csv(file_path, sep=',')
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    num_validation_samples = int(len(df_shuffled) * validation_split)

    validation_set = df_shuffled.iloc[:num_validation_samples]
    train_set = df_shuffled.iloc[num_validation_samples:]

    train_file_path = file_path.replace('.csv', '_train.csv')
    validation_file_path = file_path.replace('.csv', '_val.csv')

    train_set.to_csv(train_file_path, index=False)
    validation_set.to_csv(validation_file_path, index=False)

    print(f"Total rows in dataset: {len(df)}")
    print(f"Training set saved to: {train_file_path} with {len(train_set)} rows.")
    print(f"Validation set saved to: {validation_file_path} with {len(validation_set)} rows.")


if __name__ == "__main__":
    split_csv_train_validation(os.path.join(DATA_ROOT, "motion.csv"), validation_split=0.25)