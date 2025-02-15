import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE
import seaborn as sns

from Utils.constants import DATASETS_FOLDER_PATH, DATA_PREPROCESSED_FOLDER_PATH

def load_data_from_folder() -> pd.DataFrame:
    files = os.listdir(DATASETS_FOLDER_PATH)
    csv_files = [file for file in files if file.endswith('.csv')]
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(DATASETS_FOLDER_PATH, csv_file))
        df['file_name'] = csv_file
        dfs.append(df)
    concatenated_df = pd.concat(dfs, ignore_index=True)
    return concatenated_df

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    null_column = df.isnull().sum()
    null_column.name = "null_sum"
    df.fillna(df.mean(), inplace=True)
    return df.copy()

def get_balance_plot(df: pd.DataFrame, class_column_name:str) -> pyplot:
    classes_counts = df[class_column_name].value_counts()
    plt.bar(classes_counts.index, classes_counts.values)
    plt.xlabel(class_column_name)
    plt.ylabel('Count')
    plt.xticks(classes_counts.index, [f'Not {class_column_name}', class_column_name])
    total = sum(classes_counts)
    for index, value in classes_counts.items():
        percentage = (value / total) * 100
        plt.text(index, value, f'{percentage:.2f}%', ha='center', va='bottom')
    return plt

def smothe_data(df: pd.DataFrame, class_column_name:str) -> pd.DataFrame:
    smote = SMOTE(random_state=42)
    vY = df[class_column_name:]
    mX = df.copy().drop([class_column_name])
    x_balanced, y_balanced = smote.fit_resample(mX, vY)
    df_balanced = pd.DataFrame(np.concatenate((x_balanced, y_balanced), axis=0))
    return df_balanced

def get_pairplot(df: pd.DataFrame, class_column_name:str) -> sns.PairGrid:
    sns.set(style="ticks")
    pairplot = sns.pairplot(df, hue=class_column_name, palette={0: "red", 1: "blue"})
    pairplot.fig.subplots_adjust(top=0.95)
    pairplot.fig.suptitle('Pairplot with Target Classes')
    return pairplot

def get_corraletion_metrix(df: pd.DataFrame) -> pd.DataFrame:
    correlation_matrix_norm = df.corr()
    return sns.heatmap(correlation_matrix_norm.abs(), cmap='YlGnBu', annot=True)

def get_violinplot(df: pd.DataFrame) -> None:
    df = df.copy()
    df = df.sub(df.min()) / (df.max() - df.min())
    palette = {0.0: 'red', 1.0: 'blue'}
    fig, axs = plt.subplots(nrows=1, ncols=len(df.columns) - 1, figsize=(15, 5), sharey=True)
    sns.set(style="whitegrid")
    for i, column in enumerate(df.columns[:-1]):
        sns.violinplot(x='Potability', y=column, data=df, ax=axs[i], palette=palette, legend=False,
                       hue="Potability")
        axs[i].set_title(column)
    plt.tight_layout()
    plt.show()

# Functions after data preprocessing stage
def load_preprocessed_data_from_folder() -> pd.DataFrame:
    # Specify the folder path where the preprocessed data files are located
    folder_path = DATA_PREPROCESSED_FOLDER_PATH
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    # List all files in the folder
    files = os.listdir(folder_path)
    # Filter out only the 'preprocessed_data.csv' file
    preprocessed_data_file = "preprocessed_data.csv"
    if preprocessed_data_file not in files:
        raise FileNotFoundError(f"The file '{preprocessed_data_file}' does not exist in the folder.")
    # Load the 'preprocessed_data.csv' file into a pandas DataFrame
    file_path = os.path.join(folder_path, preprocessed_data_file)
    df = pd.read_csv(file_path)
    return df

def save_preprocessed_data_to_folder(data: pd.DataFrame) -> None:
    # Check if the specified folder exists
    if not os.path.exists(DATA_PREPROCESSED_FOLDER_PATH):
        os.makedirs(DATA_PREPROCESSED_FOLDER_PATH)  # Create the folder if it doesn't exist

    # Specify the file path for saving the preprocessed data
    file_path = os.path.join(DATA_PREPROCESSED_FOLDER_PATH, "preprocessed_data.csv")

    # Save the DataFrame to a CSV file
    data.to_csv(file_path, index=False, mode='w')  # Set index=False to exclude row indices from the CSV file


