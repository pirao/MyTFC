import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

import logging

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from pathlib import Path
from typing import List, Tuple, Union
import lightning as L
import time 

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('..')
sys.path.append('../..')
from transforms.tfc_utils import *
from scipy.fft import fft, fftfreq, ifft 

########################################
########################################
# ETL for UCI dataset
########################################
########################################

def load_uci_dataset(path, dataset_type):
    """
    Load the UCI dataset from text files in the specified path.

    Parameters:
    - path (str): The directory path where the dataset text files are located.
    - dataset_type (str): The type of dataset to load, either 'train' or 'test'.

    Returns:
    - list of pandas.DataFrame: A list of DataFrames containing all the data from the text files, 
      with additional columns for sensor names, subject numbers, and experiment numbers.

    Raises:
    - ValueError: If dataset_type is not 'train' or 'test'.

    Note:
    - The function assumes that the subject data is located in '../data/uci/subject_train.txt' 
      or '../data/uci/subject_test.txt' based on the dataset_type.
    """
    # Load each file into a dataframe
    files = [f for f in os.listdir(path) if f.endswith('.txt')]
    dataframes = []
    
    # Load subject data based on the dataset type
    if dataset_type == 'train':
        subject_dataset = pd.read_csv('../data/uci/raw/subject_train.txt', delim_whitespace=True, header=None, names=['subject'])
    elif dataset_type == 'test': 
        subject_dataset = pd.read_csv('../data/uci/raw/subject_test.txt', delim_whitespace=True, header=None, names=['subject'])
    else:
        raise ValueError('Use either "train" or "test" as dataset_type')                 
    
    # Load each sensor data file and combine with subject data
    for file in tqdm(files, total=len(files), desc=f'Loading UCI dataset from {dataset_type} folder'):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        sensor_name = file.replace('_test.txt', '').replace('_train.txt', '')  # Removing the _test and _train suffix from the filename
        df['SensorName'] = sensor_name
        df['SubjectNumber'] = subject_dataset['subject']
        df['ExperimentNumber'] = df.reset_index().index
        dataframes.append(df)
    
    return dataframes


def transform_dataframes(dataframes):
    """
    Transforms a list of dataframes by creating 'Set' columns for original columns 0-59
    and columns 60-119, renaming them to 0-59, and concatenating the results.

    Parameters:
    - dataframes (list of pd.DataFrame): List of dataframes to transform.

    Returns:
    - list of pd.DataFrame: List of transformed dataframes.
    """
    combined_dfs = []
    
    for df in tqdm(dataframes, desc='Transforming dataframes'):
        # Common columns to retain
        common_columns = ['SensorName', 'SubjectNumber', 'ExperimentNumber']
        if 'class' in df.columns:
            common_columns.append('class')

        # Create and transform subsets
        def transform_subset(start, end, set_value):
            subset = df.iloc[:, start:end].copy()
            subset.columns = map(str, range(60))
            subset['Set'] = set_value
            for col in common_columns:
                subset[col] = df[col]
            return subset

        df_0_59 = transform_subset(0, 60, 1)
        df_60_119 = transform_subset(60, 120, 2)

        # Concatenate the two DataFrames
        df_combined = pd.concat([df_0_59, df_60_119], ignore_index=True)
        combined_dfs.append(df_combined)
    
    return combined_dfs



def rename_columns(dataframes):
    """
    Rename the numerical columns of each DataFrame in the list to include the SensorName,
    drop the SensorName column (last column), and retain the 'class' column if it exists.

    Parameters:
    - dataframes (list of pandas.DataFrame): List of DataFrames to process.

    Returns:
    - list of pandas.DataFrame: List of DataFrames with renamed columns.
    """
    renamed_dataframes = []
    
    for df in tqdm(dataframes, desc='Renaming columns'):
        # Check if 'class' column exists
        if 'class' in df.columns:
            y = df['class']
            df = df.drop(['class'], axis=1)
            has_class = True
        else:
            has_class = False

        # Get the sensor name from the last column and drop it
        sensor_name = df.iloc[0, -1]
        df = df.drop(columns=df.columns[-1])
        
        # Create new column names for the numerical columns
        num_columns_to_rename = df.shape[1]
        new_column_names = [f"{sensor_name}_{i}" for i in range(num_columns_to_rename)]
        
        # Assign new column names
        df.columns = new_column_names
    
        # If 'class' column existed, add it back to the DataFrame
        if has_class:
            df = pd.concat([df, y], axis=1)
        
        renamed_dataframes.append(df)
        
    return renamed_dataframes


def standardize_columns(df, rename_dict):
    """
    Rename parts of the column names in the DataFrame according to the provided dictionary.

    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns need to be renamed.
    - rename_dict (dict): A dictionary with old parts of the column names as keys and new parts as values.

    Returns:
    - pd.DataFrame: DataFrame with renamed columns.
    """
    def rename_column(col):
        for old, new in rename_dict.items():
            if old in col:
                return col.replace(old, new)
        return col
    
    df = df.rename(columns=rename_column)
    return df

def plot_class_distribution(train_df, val_df, test_df):
    """
    Plot the class distribution of the provided DataFrames on separate subplots.

    Args:
        train_df (DataFrame): DataFrame containing training data with a 'class' column.
        val_df (DataFrame): DataFrame containing validation data with a 'class' column.
        test_df (DataFrame): DataFrame containing test data with a 'class' column.

    Returns:
        None

    Examples:
        plot_class_distribution(train_df, val_df, test_df)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    train_df['class'].value_counts().sort_index().plot(kind='bar', ax=axes[0])
    axes[0].set_title('Class Distribution in Training Set')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')

    val_df['class'].value_counts().sort_index().plot(kind='bar', ax=axes[1])
    axes[1].set_title('Class Distribution in Validation Set')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')

    test_df['class'].value_counts().sort_index().plot(kind='bar', ax=axes[2])
    axes[2].set_title('Class Distribution in Test Set')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.show()
    
    
    
#####################################################
#####################################################
# UCI Dataset Class
#####################################################
#####################################################

class UCIDataset_4_TFC(Dataset):
    def __init__(self, parquet_file_path: Union[Path, str],
                 feature_column_prefixes: list = ("accel-x",
                                                  "accel-y",
                                                  "accel-z",
                                                  "gyro-x",
                                                  "gyro-y",
                                                  "gyro-z",),
                 target_column: str = "class",
                 flatten: bool = False,
                 
                 transform=None,
                 dt=0.02,
                 training_mode='TFC'): 
        
        self.parquet_file_path = Path(parquet_file_path)
        self.feature_column_prefixes = feature_column_prefixes
        self.target_column = target_column
        self.flatten = flatten

        logging.debug("Loading data from Parquet file")
        self.data = pd.read_parquet(self.parquet_file_path)

        self.columns_to_select = [self._list_columns_starting_with(self.data, prefix) for prefix in self.feature_column_prefixes]
        
        ########## Extra I added based on seismic.py ##########
        self.transform = transform
        self.training_mode = training_mode
        self.dt = dt
            

    @staticmethod
    def _list_columns_starting_with(df: pd.DataFrame, prefix: str) -> List[str]:
        return [column for column in df.columns if column.startswith(prefix)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        
        # Initializing the data and freq data
        data = np.zeros(shape=(len(self.columns_to_select), len(self.columns_to_select[0])), dtype=np.float32)
        
        for channel in range(len(self.columns_to_select)):
            columns = self.columns_to_select[channel]
            if len(columns) == 0:
                raise ValueError(f"No columns found for prefix {self.feature_column_prefixes[channel]}")
            data[channel, :] = self.data.iloc[idx][columns]
        
        if self.flatten:
            data = data.flatten()
        target = int(self.data.iloc[idx][self.target_column]) -1 # Subtracting 1 to make the target 0-based 
        
        
        if len(data.shape) != 2:
            raise ValueError(f"Input time_sample must have 2 dimensions (num_sensors, num_timesteps), but got {data.shape}.")
        
        # If we are working with TFC and want to apply the data augmentation
        if self.transform and self.training_mode == 'TFC':
            # Compute the normalized amplitude spectrum
            norm_amplitude_spectrum_dataset = compute_half_spectrum_for_dataset(data, self.dt)
            norm_full_amplitude_spectrum_dataset = reconstruct_full_spectrum(norm_amplitude_spectrum_dataset)
            
            # Apply transformations on the time series and half spectrum
            _, data_aug_time, norm_amplitude_spectrum_aug = self.transform(data, norm_amplitude_spectrum_dataset) 
            
            # Transforming from augmented half spectrum to full spectrum 
            norm_full_amplitude_spectrum_aug_dataset = reconstruct_full_spectrum(norm_amplitude_spectrum_aug)
            
            # Checking time dimension
            if data_aug_time.shape[-1] != norm_full_amplitude_spectrum_aug_dataset.shape[-1]:
                print(f"data_aug_time shape: {data_aug_time.shape}")
                print(f"norm_amplitude_spectrum_aug shape: {norm_full_amplitude_spectrum_aug_dataset.shape}")
                raise ValueError("The time dimension of data_aug_time should be equal to norm_amplitude_spectrum_aug for full spectrum.")            
            
            return data, data_aug_time, norm_full_amplitude_spectrum_dataset, norm_full_amplitude_spectrum_aug_dataset, target
        else:
            return data, target
    
    
class UCIDataModule_4_TFC(L.LightningDataModule):
    """
    Encapsulates the data loading and processing for the HAR dataset using Parquet files.
    """

    def __init__(self,root_data_dir: Union[Path, str],
                 feature_column_prefixes: list = ("accel-x","accel-y","accel-z","gyro-x","gyro-y","gyro-z"),
        target_column: str = "class",
        flatten: bool = False,
        batch_size: int = 32,
        transform=None,
        training_mode='TFC'):
        
        super().__init__()
        self.root_data_dir = Path(root_data_dir)
        self.feature_column_prefixes = feature_column_prefixes
        self.target_column = target_column
        self.flatten = flatten
        self.batch_size = batch_size
        self.parquet_files = {
            "train": os.path.join(self.root_data_dir, "train.parquet"),
            "validation": os.path.join(self.root_data_dir, "validation.parquet"),
            "test": os.path.join(self.root_data_dir, "test.parquet"),
        }
        
        # Extra I added for TFC
        self.transform = transform
        self.training_mode = training_mode

    def _get_dataset_dataloader(self, path: Path, shuffle: bool) -> DataLoader:
        """Create a UCIDataset object and return a DataLoader object."""
        dataset = UCIDataset_4_TFC(
            parquet_file_path=path,
            feature_column_prefixes=self.feature_column_prefixes,
            target_column=self.target_column,
            flatten=self.flatten,
            transform=self.transform,
            training_mode=self.training_mode
        )
    
        
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=3,  # Doesnt work if num_workers > 0
            pin_memory=True,
            drop_last = True,
            persistent_workers=True
        )
        
        return dataloader
    
    def train_dataloader(self):
        """Returns a DataLoader for the training dataset."""
        return self._get_dataset_dataloader(self.parquet_files["train"], shuffle=True)
    
    def val_dataloader(self):
        """Returns a DataLoader for the validation dataset."""
        return self._get_dataset_dataloader(self.parquet_files["validation"], shuffle=False)
    
    def test_dataloader(self):
        """Returns a DataLoader for the test dataset."""
        return self._get_dataset_dataloader(self.parquet_files["test"], shuffle=False)