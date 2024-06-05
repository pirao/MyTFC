import glob
import os
import urllib.request
import zipfile
import lightning as L
import numpy as np
import tifffile as tiff
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from functools import lru_cache
import random
class F3SeismicDataset(Dataset):
    """
    A custom dataset class for the F3 seismic dataset.

    Parameters
    ----------
        data_dir: str
            The directory path where the data files are located.
        labels_dir: str
            The directory path where the label files are located.
        transform: callable, optional
            A function/transform that takes in a np.array representing the sample feattures and returns a transformed version of this sample
            Default is None.
        target_transform: callable, optional
            A function/transform that takes in a np.array representing the sample label and returns a transformed version of this label
            Default is None.
    """

    def __init__(self, data_dir, labels_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = glob.glob(data_dir + "/*.tif")

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
            int
                The total number of samples.
        """
        return int(len(self.data))

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Parameters
        ----------
            idx: int
                The index of the sample to retrieve.

        Returns
        -------
            tuple
                A tuple containing the image and label of the sample. 
                In case transform/target_transform methods are provided, it returns the trasformed version of the image/label.
        """
        img, label = self._read_img_label(idx)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    @lru_cache(maxsize=None)
    def _read_img_label(self, idx):
        """
        Read the sample image and its respective label from files. 

        Parameters
        ----------
            idx: int
                The index of the sample to retrieve.

        Returns
        -------
            tuple
                A tuple containing the image and label of the sample. 
        """        
        img = tiff.imread(self.data[idx])
        img_base_name = os.path.basename(self.data[idx]).split(".")[0]

        label = np.array(
            Image.open(os.path.join(self.labels_dir, img_base_name + ".png"))
        )
        img = self._pad(img, [255, 701])
        label = self._pad(label, [255, 701])
        return img, label

    def _pad(self, x, target_size):
        """
        Pads the image to achieve the target size. 

        Parameters
        ----------
            x: np.array
                The image to pad.
            target_size: (h, w)
                The target height and width.

        Returns
        -------
            padded_img
                The padded image. 
        """        
        h, w = x.shape[:2]
        pad_h = max(0, target_size[0] - h)
        pad_w = max(0, target_size[1] - w)
        if len(x.shape) == 2:
            padded = np.pad(x, ((0, pad_h), (0, pad_w)), mode="reflect")
            padded = np.expand_dims(padded, axis=2)
        else:
            padded = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            padded = padded.astype(float)

        padded_img = np.transpose(padded, (2, 0, 1))

        return padded_img

class F3SeismicDataModule(L.LightningDataModule):
    """
    A custom dataset module for the F3 seismic dataset.

    Parameters
    ----------
        root_dir: str
            The root directory path where the data files are located.
        batch_size: int
            The batch size for the DataLoaders.
        shuffle_dataset_indices: bool
            If true, the indices of the train, validation, and test datasets will be shuffled. 
            They are only shuffled once, at the DataModule initialization.
        transform: callable, optional
            A function/transform that takes in a np.array representing the sample feattures and returns a transformed version of this sample.
            This function will be employed on the train, validation and test datasets.
            Default is None.
        target_transform: callable, optional
            A function/transform that takes in a np.array representing the sample label and returns a transformed version of this label
            This function will be employed on the train, validation and test datasets.
            Default is None.
    """    
    def __init__(self, 
                 root_dir, 
                 batch_size=32, 
                 num_workers=8, 
                 shuffle_dataset_indices = True,
                 transform=None, 
                 target_transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_dataset_indices = shuffle_dataset_indices
        self.transform = transform
        self.target_transform = target_transform
        self.zip_file = root_dir + "f3.zip"
        self.setup()

    def setup(self, stage:str = None):
        # Check if root dir exists
        if not os.path.exists(self.root_dir):
            print(f"Creating the root directory: [{self.root_dir}]")
            os.makedirs(self.root_dir)

        # Check if f3.zip file exists. If not, download it
        if not os.path.exists(self.zip_file):
            print(f"Could not find the zip file [{self.zip_file}]")
            print(f"Trying to download it.")
            url = "https://www.ic.unicamp.br/~edson/disciplinas/mo436/2024-1s/data/f3.zip"
            urllib.request.urlretrieve(url, self.zip_file)

        # Check if f3/ folder exists. If not, unzip the f3.zip file
        if not os.path.exists(self.root_dir + "/f3"):
            with zipfile.ZipFile(self.zip_file, "r") as zip_ref:
                zip_ref.extractall(self.root_dir)
            print("F3 Dataset extracted from {self.zip_file}")

        self.data_dir   = os.path.join(self.root_dir, "f3", "images")
        self.labels_dir = os.path.join(self.root_dir, "f3", "annotations")

        self.train_dataset = F3SeismicDataset(
            self.data_dir + "/train", self.labels_dir + "/train", 
            transform=self.transform, target_transform=self.target_transform
        )
        self.val_dataset = F3SeismicDataset(
            self.data_dir + "/val", self.labels_dir + "/val",
            transform=self.transform, target_transform=self.target_transform
        )
        self.test_dataset = F3SeismicDataset(
            self.data_dir + "/test", self.labels_dir + "/test",
            transform=self.transform, target_transform=self.target_transform
        )

        self.train_indices = list(range(len(self.train_dataset)))
        self.val_indices   = list(range(len(self.val_dataset)))
        self.test_indices  = list(range(len(self.test_dataset)))
        if self.shuffle_dataset_indices:
            random.shuffle(self.train_indices)
            random.shuffle(self.val_indices)
            random.shuffle(self.test_indices)

    def train_dataloader(self, cap: float = 1.0, drop_last=False):
        """
        Retrieves the training dataloader. 

        Parameters
        ----------
            cap: float
                Percentage of the dataset to be used in the dataloader. 
                Ex: cap=0.4 implies only the first 40% of the dataset items will be retrieved by the dataloader.
            drop_last: Bool
                If True, the last batch is dropped by the dataloader (ensures all batches have the same size)

        Returns
        -------
            dataloader
                The training dataloader. 
        """        
        N = int(cap*len(self.train_dataset))
        dataset = torch.utils.data.Subset(self.train_dataset, self.train_indices[0:N])
        return DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=True, drop_last=drop_last,
            num_workers = self.num_workers
        )

    def val_dataloader(self, cap: float = 1.0, drop_last=False):
        """
        Retrieves the validation dataloader. 

        Parameters
        ----------
            cap: float
                Percentage of the dataset to be used in the dataloader. 
                Ex: cap=0.4 implies only the first 40% of the dataset items will be retrieved by the dataloader.
            drop_last: Bool
                If True, the last batch is dropped by the dataloader (ensures all batches have the same size)

        Returns
        -------
            dataloader
                The validation dataloader. 
        """        
        N = int(cap*len(self.val_dataset))
        dataset = torch.utils.data.Subset(self.val_dataset, self.val_indices[0:N])
        return DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=False, drop_last=drop_last,
            num_workers = self.num_workers
        )

    def test_dataloader(self, cap: float = 1.0, drop_last=False):
        """
        Retrieves the test dataloader. 

        Parameters
        ----------
            cap: float
                Percentage of the dataset to be used in the dataloader. 
                Ex: cap=0.4 implies only the first 40% of the dataset items will be retrieved by the dataloader.
            drop_last: Bool
                If True, the last batch is dropped by the dataloader (ensures all batches have the same size)

        Returns
        -------
            dataloader
                The test dataloader. 
        """        
        N = int(cap*len(self.test_dataset))
        dataset = torch.utils.data.Subset(self.test_dataset, self.test_indices[0:N])
        return DataLoader(
            dataset, batch_size=self.batch_size, 
            shuffle=False, drop_last=drop_last,
            num_workers = self.num_workers
        )
