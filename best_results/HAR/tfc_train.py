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
from data_modules.har import *
from tfc_utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)


########################################################################################



def load_pretrained_backbone(pretrained_backbone_checkpoint_filename,global_config_file):
    backbone = TFC_Backbone(global_config_file)
    backbone.load_state_dict(torch.load(pretrained_backbone_checkpoint_filename))
    return backbone


def build_downstream_datamodule(global_config_file, batch_size) -> L.LightningDataModule:
    
    # Build the transform object
    tfc_transforms = TFC_transforms(global_config_file, verbose=False)
    
    return HarDataModule(root_data_dir="../../data/har", 
                         batch_size=batch_size,
                         flatten = False, 
                         target_column = "standard activity code",
                         training_mode='TFC',
                         transform=tfc_transforms)


def build_downstream_model(global_config_file, backbone, prediction_head) -> L.LightningModule:
    return TFC_Combined_Model(global_config=global_config_file,
                              backbone=backbone,
                              projector_head=prediction_head).to(device)


def build_lightning_trainer(log_dir='lightning_logs', experiment_name='tf_c', version=0, verbose=True):
    tfc_callbacks = TFCCallbacks(
        monitor='val_f1',
        min_delta=0.001,
        patience=30,
        verbose=verbose,
        log_dir=log_dir,
        experiment_name=experiment_name,
        version=version
    )
    
    callbacks = tfc_callbacks.get_callbacks()
    logger = tfc_callbacks.get_logger()
    
    return L.Trainer(
        max_epochs=200, # 80
        accelerator='gpu',
        num_sanity_val_steps=0,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        benchmark=True,
        callbacks=callbacks)
    
def delete_existing_logs(log_dir, experiment_name, version):
    log_path = os.path.join(log_dir, experiment_name, f"version_{version}")
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
        print(f"Deleted existing logs in: {log_path}")
        
def downstream_save_backbone_weights(pretext_model, checkpoint_filename):
    print(f"Saving backbone pretrained weights at {checkpoint_filename}")
    torch.save(pretext_model.backbone.state_dict(), checkpoint_filename)

#########################################
# Main function
#########################################        

def main(SSL_technique_prefix, freeze=True, batch_size = 128):
    
    # File containing all relevant hyperparameters
    global_config_file = GlobalConfigFile(batch_size)
    
    log_dir = "lightning_logs"
    experiment_name = SSL_technique_prefix
    version = 420
    delete_existing_logs(log_dir, experiment_name, version)
    
    # Getting the pretrained backbone
    pretrained_backbone_checkpoint_filename = f"./{SSL_technique_prefix}_pretrained_backbone_weights.pth"
    backbone = load_pretrained_backbone(pretrained_backbone_checkpoint_filename,global_config_file)
    
    # Build the downstream model, the downstream datamodule, and the trainer
    downstream_datamodule = build_downstream_datamodule(batch_size=batch_size,
                                                        global_config_file=global_config_file)
    
    downstream_model = build_downstream_model(backbone=backbone, 
                                              prediction_head=None, # Use the default head
                                              global_config_file=global_config_file)
    
    # Checking if just the backbone is frozen and not the prediction head 
    if freeze:
        downstream_model.freeze_backbone() 
    else:
        downstream_model.unfreeze_backbone()
    
    
    lightning_trainer = build_lightning_trainer(log_dir=log_dir, 
                                                experiment_name=experiment_name, 
                                                verbose=True,version=version)

    # Fit the downstream model using the downstream datamodule
    lightning_trainer.fit(downstream_model,downstream_datamodule) 
    
    # Save the downstream weights   
    output_filename = f"./{SSL_technique_prefix}_downstream_model.pth"
    downstream_save_backbone_weights(downstream_model, output_filename)
    print(f"Downstream weights saved at: {output_filename}")
    
    return downstream_model, lightning_trainer


if __name__ == "__main__":
    SSL_technique_prefix = "TF_C"
    model, downstream_lightning_trainer = main(SSL_technique_prefix, freeze=True,batch_size=8)
    plot_model_metrics(model)