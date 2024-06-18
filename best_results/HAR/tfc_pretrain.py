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
from data_modules.uci_4_tfc import *
from transforms.tfc_utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)


########################################################################################

def build_pretext_datamodule(global_config_file,batch_size=32, root_data_dir='../../data/uci/preprocessed/') -> L.LightningDataModule:
    
    # Build the transform object
    tfc_transforms = TFC_transforms(global_config_file, verbose=False)
    
    # Create the datamodule
    return UCIDataModule_4_TFC(root_data_dir=root_data_dir,
                           batch_size=batch_size,
                           flatten=False,
                           target_column='class',
                           training_mode='TFC',
                           transform=tfc_transforms)
    
    
def build_pretext_model(global_config_file) -> L.LightningModule:
    return TFC_Backbone(global_config=global_config_file)

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
    
    
    
def pretext_save_backbone_weights(pretext_model, save_dir, SSL_technique_prefix):
    """
    Saves the backbone pretrained weights at the specified directory.

    Args:
        pretext_model (nn.Module): The pretext model whose weights are to be saved.
        save_dir (str): The directory where the .pth file will be saved.
        SSL_technique_prefix (str): The prefix to use in the filename.
    """
    pth_dir = os.path.join(save_dir, 'pth_files')
    os.makedirs(pth_dir, exist_ok=True)
    checkpoint_filename = os.path.join(pth_dir, f"{SSL_technique_prefix}_pretrained_backbone_weights.pth")
    print(f"Saving backbone pretrained weights at {checkpoint_filename}")
    torch.save(pretext_model.state_dict(), checkpoint_filename)
    

        
#########################################
# Main function
#########################################        

# More epochs = better performance. I used up to 100 epochs for the best results, until my PC crashed from GPU overheating....
def main(SSL_technique_prefix,batch_size=128):
    
    # File containing all relevant hyperparameters
    global_config_file = GlobalConfigFile(batch_size)
    
    # Adding information to create paths to save the logs
    log_dir = "lightning_logs"
    experiment_name = SSL_technique_prefix
    version = "Backbone_Pretraining"
    
    # Build the pretext model, the pretext datamodule, and the trainer
    pretext_datamodule = build_pretext_datamodule(global_config_file, batch_size=batch_size)
    pretext_model = build_pretext_model(global_config_file)
    
    lightning_trainer, experiment_path = build_lightning_trainer(log_dir=log_dir, 
                                                                 experiment_name=experiment_name, 
                                                                 verbose=True,
                                                                 version=version,
                                                                 SSL_technique_prefix=SSL_technique_prefix)

    # Fit the pretext model using the pretext_datamodule
    lightning_trainer.fit(pretext_model, pretext_datamodule)
    
    # Save the pretrain backbone weights
    pretext_save_backbone_weights(pretext_model, experiment_path, SSL_technique_prefix)
    print(f"Pretrained weights saved at: {os.path.join(experiment_path, 'pth_files', f'{SSL_technique_prefix}_pretrained_backbone_weights.pth')}")
    
    return pretext_model, lightning_trainer


if __name__ == "__main__":
    SSL_technique_prefix = "TF_C"
    model, pretrain_lightning_trainer = main(SSL_technique_prefix)