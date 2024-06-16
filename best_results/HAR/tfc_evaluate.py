import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../transforms/')
sys.path.append('../../TFC_Configs/')
sys.path.append('../../models/')
sys.path.append('../../data_modules/')
sys.path.append('../../tfc_utils/') 

import torch
import numpy as np
import lightning as L
import shutil

import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger

from config_files.TFC_Configs import *
from transforms.tfc_augmentations import *
from transforms.tfc_utils import *
from models.tfc import *
from data_modules.uci import *
from tfc_utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

from torchmetrics import JaccardIndex

def evaluate_model(model, dataset_dl):
    # Inicialize JaccardIndex metric
    jaccard = JaccardIndex(task="multiclass", num_classes=6)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # For each batch, compute the predictions and compare with the labels.
    for X, y in dataset_dl:
        # Move the model, data and metric to the GPU if available
        model.to(device)
        X = X.to(device)
        y = y.to(device)
        jaccard.to(device)

        logits = model(X.float())
        predictions = torch.argmax(logits, dim=1)
        jaccard(predictions, y)
    # Return a tuple with the number of correct predictions and the total number of predictions
    return (float(jaccard.compute().to("cpu")))

def report_IoU(model, dataset_dl, prefix=""):
    iou = evaluate_model(model, dataset_dl)
    print(prefix + " IoU = {:0.4f}".format(iou))