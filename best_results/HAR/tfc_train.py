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
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import TensorBoardLogger

from config_files.TFC_Configs import *
from transforms.tfc_augmentations import *
from transforms.tfc_utils import *
from models.tfc import *
from data_modules.har_4_tfc import *
from transforms.tfc_utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)


########################################################################################



def load_pretrained_backbone(pretrained_backbone_checkpoint_filename,global_config_file):
    backbone = TFC_Backbone(global_config_file)
    backbone.load_state_dict(torch.load(pretrained_backbone_checkpoint_filename))
    return backbone


def build_downstream_datamodule(global_config_file, batch_size,root_data_dir="../../data/har") -> L.LightningDataModule:
    
    # Build the transform object
    tfc_transforms = TFC_transforms(global_config_file, verbose=False)
    
    return HarDataModule(root_data_dir=root_data_dir, 
                         batch_size=batch_size,
                         flatten = False, 
                         target_column = "standard activity code",
                         transform=tfc_transforms)


def build_downstream_model(global_config_file, backbone, prediction_head) -> L.LightningModule:
    return TFC_Combined_Model(global_config=global_config_file,
                              backbone=backbone,
                              projector_head=prediction_head).to(device)


def build_lightning_trainer(log_dir='lightning_logs', experiment_name='tf_c', 
                            version=0, verbose=True, 
                            max_epochs=3,min_delta=0.01,patience=3,
                            monitor='train_loss_total',SSL_technique_prefix='TF_C') -> L.Trainer: 

    tfc_callbacks = TFCCallbacks(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        verbose=verbose,
        log_dir=log_dir,
        experiment_name=experiment_name,
        version=version,
        SSL_technique_prefix=SSL_technique_prefix
    )
    
    callbacks = tfc_callbacks.get_callbacks()
    logger = tfc_callbacks.get_logger()
    
    # Ensure necessary directories exist
    experiment_path = tfc_callbacks.experiment_path
    pth_path = os.path.join(experiment_path, 'pth_files')
    os.makedirs(pth_path, exist_ok=True)
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        num_sanity_val_steps=0,
        logger=logger,
        log_every_n_steps=1,
        benchmark=True,
        callbacks=callbacks)
    
    
    return trainer, experiment_path

        
def downstream_save_weights(downstream_model, save_dir, SSL_technique_prefix):
    """
    Saves the backbone pretrained weights at the specified directory.

    Args:
        downstream_model (nn.Module): The downstream model whose weights are to be saved (both the backbone and prediction head).
        save_dir (str): The directory where the .pth file will be saved.
        SSL_technique_prefix (str): The prefix to use in the filename.
    """
    pth_dir = os.path.join(save_dir, 'pth_files')
    os.makedirs(pth_dir, exist_ok=True)
    checkpoint_filename = os.path.join(pth_dir, f"{SSL_technique_prefix}_downstream_weights.pth")
    print(f"Saving downstream weights at {checkpoint_filename}")
    torch.save(downstream_model.state_dict(), checkpoint_filename)

#########################################
# Main function
#########################################        

def main(SSL_technique_prefix, freeze=True, batch_size = 2):
    
    # File containing all relevant hyperparameters
    global_config_file = GlobalConfigFile(batch_size)
    
    log_dir = "lightning_logs"
    experiment_name = SSL_technique_prefix
    version = 'Downstream_Task'
    
    # Getting the pretrained backbone
    pretrained_backbone_checkpoint_filename = f"./{log_dir}/{SSL_technique_prefix}/Backbone_Pretraining/pth_files/TF_C_pretrained_backbone_weights.pth"
    backbone = load_pretrained_backbone(pretrained_backbone_checkpoint_filename,global_config_file)
    
    downstream_datamodule = build_downstream_datamodule(batch_size=batch_size,global_config_file=global_config_file)
    downstream_model = build_downstream_model(backbone=backbone, 
                                              prediction_head=None, # Use the default head
                                              global_config_file=global_config_file)
    
    # Checking if just the backbone is frozen and not the prediction head 
    if freeze:
        downstream_model.freeze_backbone() 
    else:
        downstream_model.unfreeze_backbone()
    
    
    lightning_trainer, experiment_path = build_lightning_trainer(log_dir=log_dir, 
                                                                 experiment_name=experiment_name, 
                                                                 verbose=True,
                                                                 version=version,
                                                                 SSL_technique_prefix=SSL_technique_prefix)

    # Fit the downstream model using the downstream datamodule
    lightning_trainer.fit(downstream_model,downstream_datamodule) 
  
    downstream_save_weights(downstream_model,  experiment_path, SSL_technique_prefix)
    print(f"Downstream weights saved at: {os.path.join(experiment_path, 'pth_files', f'{SSL_technique_prefix}_downstream_weights.pth')}")
    
    return downstream_model, lightning_trainer

# For best results, I recommend using Freeze = False and a large number of epochs
if __name__ == "__main__":
    SSL_technique_prefix = "TF_C"
    model, downstream_lightning_trainer = main(SSL_technique_prefix, freeze=False,batch_size=2)