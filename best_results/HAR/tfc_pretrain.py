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


########################################################################################

def build_pretext_datamodule(global_config_file,batch_size=32) -> L.LightningDataModule:
    
    # Build the transform object
    tfc_transforms = TFC_transforms(global_config_file, verbose=False)
    
    # Create the datamodule
    return UCIDataModule(root_data_dir='../../data/uci/preprocessed/',
                           batch_size=batch_size,
                           flatten=False,
                           target_column='class',
                           training_mode='TFC',
                           transform=tfc_transforms)
    
def build_pretext_model(global_config_file) -> L.LightningModule:
    return TFC_Combined_Model(global_config=global_config_file).to(device)


def build_lightning_trainer(log_dir='lightning_logs', experiment_name='tf_c', version=0, verbose=True):
    tfc_callbacks = TFCCallbacks(
        monitor='val_loss_total',
        min_delta=0.01,
        patience=5,
        verbose=verbose,
        log_dir=log_dir,
        experiment_name=experiment_name,
        version=version
    )
    
    callbacks = tfc_callbacks.get_callbacks()
    logger = tfc_callbacks.get_logger()
    
    return L.Trainer(
        max_epochs=100, # 80
        accelerator='gpu',
        num_sanity_val_steps=0,
        logger=logger,
        log_every_n_steps=1,
        benchmark=True,
        callbacks=callbacks)
    
def pretext_save_backbone_weights(pretext_model, checkpoint_filename):
    print(f"Saving backbone pretrained weights at {checkpoint_filename}")
    torch.save(pretext_model.backbone.state_dict(), checkpoint_filename)
    
def delete_existing_logs(log_dir, experiment_name, version):
    log_path = os.path.join(log_dir, experiment_name, f"version_{version}")
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
        print(f"Deleted existing logs in: {log_path}")
        
        
#########################################
# Main function
#########################################        

def main(SSL_technique_prefix,batch_size=128):
    
    # File containing all relevant hyperparameters
    global_config_file = GlobalConfigFile(batch_size)
    
    log_dir = "lightning_logs"
    experiment_name = SSL_technique_prefix
    version = 420
    delete_existing_logs(log_dir, experiment_name, version)
    
    # Build the pretext model, the pretext datamodule, and the trainer
    pretext_datamodule = build_pretext_datamodule(global_config_file, batch_size=batch_size)
    pretext_model = build_pretext_model(global_config_file)
    lightning_trainer = build_lightning_trainer(log_dir=log_dir, experiment_name=experiment_name, verbose=True,version=version)

    # Fit the pretext model using the pretext_datamodule
    lightning_trainer.fit(pretext_model, pretext_datamodule)
    
    # Save the pretrain backbone weights
    output_filename = f"./{SSL_technique_prefix}_pretrained_backbone_weights.pth"
    pretext_save_backbone_weights(pretext_model, output_filename)
    print(f"Pretrained weights saved at: {output_filename}")
    
    return pretext_model, lightning_trainer


if __name__ == "__main__":
    SSL_technique_prefix = "TF_C"
    model, pretrain_lightning_trainer = main(SSL_technique_prefix)
    plot_model_metrics(model)