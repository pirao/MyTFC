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

import lightning.pytorch as L

from config_files.TFC_Configs import *
from transforms.tfc_augmentations import *
from transforms.tfc_utils import *
from models.tfc import *
from data_modules.har_4_tfc import *
from transforms.tfc_utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

# Set CuDNN benchmarking to False
import torch.backends.cudnn as cudnn
cudnn.benchmark = False


########################################################################################
### - Extra Code --------------------------------------------------------------------
from torchmetrics import JaccardIndex

### Code of the professor
# def evaluate_model(model, dataset_dl):
#     # Inicialize JaccardIndex metric
#     jaccard = JaccardIndex(task="multiclass", num_classes=6)

#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # For each batch, compute the predictions and compare with the labels.
#     for X, y in dataset_dl:
#         # Move the model, data and metric to the GPU if available
#         model.to(device)
#         X = X.to(device)
#         y = y.to(device)
#         jaccard.to(device)

#         logits = model(X.float())
#         predictions = torch.argmax(logits, dim=1, keepdim=True)
#         jaccard(predictions, y)
#     # Return a tuple with the number of correct predictions and the total number of predictions
#     return (float(jaccard.compute().to("cpu")))

# def report_IoU(model, dataset_dl, prefix=""):
#     iou = evaluate_model(model, dataset_dl)
#     print(prefix + " IoU = {:0.4f}".format(iou))


# TF_C updated version to work with my data augmentations
def evaluate_model_V2(model, dataset_dl):
    """
    Evaluates the model on the given dataset and returns the Jaccard Index (IoU).

    Args:
        model (nn.Module): The model to evaluate.
        dataset_dl (DataLoader): The dataloader for the dataset.

    Returns:
        float: The Jaccard Index (IoU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize JaccardIndex metric and move it to the correct device
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=6).to(device)

    # Move the model to the correct device
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Ensure no gradients are calculated during evaluation
    with torch.no_grad():
        for batch in dataset_dl:
            # Unpack the batch based on the structure of your dataset
            time_data, time_aug_data, freq_data, freq_aug_data, labels = batch
            time_data = time_data.to(device)
            freq_data = freq_data.to(device)
            labels = labels.to(device)

            # Forward pass
            with torch.no_grad():  # Ensure no gradients are calculated
                logits, _ = model(time_data, freq_data)
                predictions = torch.argmax(logits, dim=1)
            
            # Update the Jaccard Index
            jaccard.update(predictions, labels)

    # Compute and return the IoU
    return float(jaccard.compute().to("cpu"))

def report_IoU_V2(model, dataset_dl, prefix=""):
    iou = evaluate_model_V2(model, dataset_dl)
    print(prefix + " IoU = {:0.4f}".format(iou))

### -------------------------------------------------------------------------------


def load_downstream_model(downstream_model_checkpoint_filename, global_config_file):
    downstream_model = TFC_Combined_Model.load_from_checkpoint(downstream_model_checkpoint_filename, global_config=global_config_file)
    return downstream_model.to(device)

def build_downstream_datamodule(global_config_file, batch_size,root_data_dir="../../data/har") -> L.LightningDataModule:
    
    # Build the transform object
    tfc_transforms = TFC_transforms(global_config_file, verbose=False)
    
    return HarDataModule(root_data_dir=root_data_dir, 
                         batch_size=batch_size,
                         flatten = False, 
                         target_column = "standard activity code",
                         transform=tfc_transforms)


def main(SSL_technique_prefix,batch_size=2):
    
    downstream_model_checkpoint_filename = f"./lightning_logs/{SSL_technique_prefix}/Downstream_Task/checkpoints/TF_C-model-epoch=2-train_loss_total=16.83.ckpt"
    global_config_file = GlobalConfigFile(batch_size=batch_size)
    downstream_model = load_downstream_model(downstream_model_checkpoint_filename = downstream_model_checkpoint_filename,
                                             global_config_file = global_config_file)

    downstream_datamodule = build_downstream_datamodule(batch_size=batch_size,
                                                        global_config_file=global_config_file)
    
    train_dl = downstream_datamodule.train_dataloader()
    val_dl   = downstream_datamodule.val_dataloader()
    test_dl  = downstream_datamodule.test_dataloader()  
    
    # # Compute and report the mIoU metric for each subset
    report_IoU_V2(downstream_model, train_dl, prefix="   Training dataset")
    report_IoU_V2(downstream_model, val_dl,   prefix=" Validation dataset")
    report_IoU_V2(downstream_model, test_dl,  prefix="       Test dataset")

if __name__ == "__main__":
    SSL_technique_prefix = "TF_C"
    main(SSL_technique_prefix)