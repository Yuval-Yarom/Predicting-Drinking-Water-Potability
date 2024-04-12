import os
import pandas as pd
from Utils.constants import DATASETS_FOLDER_PATH

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
