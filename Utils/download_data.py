import os
import zipfile
import requests
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_load_csv(dataset_name, file_name):
    # Instantiate the Kaggle API
    api = KaggleApi()

    # Replace 'username/dataset-name' with the actual dataset path on Kaggle
    dataset_path = f'{dataset_name}/{file_name}'

    # Download the dataset
    api.authenticate()  # Make sure to authenticate first
    list = api.datasets_list()
    api.dataset_download_files(dataset_path, unzip=True)

    # Find the downloaded CSV file
    files = os.listdir()
    csv_files = [file for file in files if file.endswith('.csv')]

    if len(csv_files) == 0:
        print("No CSV file found in the downloaded dataset.")
        return None

    # Load CSV file into a Pandas DataFrame
    csv_file = csv_files[0]  # Assuming only one CSV file is downloaded
    df = pd.read_csv(csv_file)

    # Remove downloaded files
    os.remove(csv_file)

    return df

if __name__ == '__main__':
    dataset_name = 'thomaskonstantin/exploring-and-predicting-drinking-water-potability/input'
    file_name = 'dataset_1.csv'

    df = download_and_load_csv(dataset_name, file_name)
    if df is not None:
        print(df.head())