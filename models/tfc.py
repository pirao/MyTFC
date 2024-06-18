import torch
import numpy as np

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import os
# import pytorch_lightning as LightningModule

import torchmetrics
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
sys.path.append('..')
from config_files.TFC_Configs import GlobalConfigFile

######################
# TFC model
#######################

class TFC_Backbone(L.LightningModule): 
    def __init__(self, global_config: GlobalConfigFile, log_metrics=True):
        super(TFC_Backbone, self).__init__()

        # Initialize metric logs
        self.log_metrics = log_metrics
        self.train_losses = []
        self.val_losses = []
        self.train_loss_time_enc = []
        self.val_loss_time_enc = []
        self.train_loss_freq_enc = []
        self.val_loss_freq_enc = []
        self.train_loss_consist = []
        self.val_loss_consist = []
        # self.initial_validation = True
        
        configs = global_config.enc_config
        loss_config = global_config.loss_config
        
        self.nt_xent_criterion = NTXentLoss(global_config).to(device)
        self.lambda_val = loss_config.lambda_val
        self.optimizer_config = global_config.optimizer_config
        
        ###################################
        # Time-based Encoder
        ###################################
        
        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(in_channels=configs.l1_input_channels, 
                      out_channels=configs.l1_output_channels, 
                      kernel_size=configs.l1_kernel_size,
                      stride=configs.l1_stride, 
                      bias=False,
                      padding=configs.l1_padding),
            nn.BatchNorm1d(configs.l1_output_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=configs.l1_maxpool1d_kernel_size, 
                         stride=configs.l1_maxpool1d_stride, 
                         padding=configs.l1_maxpool1d_padding),
            nn.Dropout(configs.l1_dropout)
        )

        self.conv_block2_t = nn.Sequential(
            nn.Conv1d(in_channels=configs.l2_input_channels, 
                      out_channels=configs.l2_output_channels, 
                      kernel_size=configs.l2_kernel_size, 
                      stride=configs.l2_stride, 
                      bias=False, 
                      padding=configs.l2_padding),
            nn.BatchNorm1d(configs.l2_output_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=configs.l2_maxpool1d_kernel_size, 
                         stride=configs.l2_maxpool1d_stride, 
                         padding=configs.l2_maxpool1d_padding)
        )

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(in_channels=configs.l3_input_channels,
                      out_channels=configs.l3_output_channels, 
                      kernel_size=configs.l3_kernel_size, 
                      stride=configs.l3_stride, 
                      bias=False, 
                      padding=configs.l3_padding),
            nn.BatchNorm1d(configs.l3_output_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=configs.l3_maxpool1d_kernel_size, 
                         stride=configs.l3_maxpool1d_stride, 
                         padding=configs.l3_maxpool1d_padding)
        )

        ###################################
        # Frequency-based Encoder
        ###################################
        
        self.conv_block1_f = nn.Sequential(
            nn.Conv1d(in_channels=configs.l1_input_channels, 
                      out_channels=configs.l1_output_channels, 
                      kernel_size=configs.l1_kernel_size,
                      stride=configs.l1_stride, 
                      bias=False,
                      padding=configs.l1_padding),
            nn.BatchNorm1d(configs.l1_output_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=configs.l1_maxpool1d_kernel_size, 
                         stride=configs.l1_maxpool1d_stride, 
                         padding=configs.l1_maxpool1d_padding),
            nn.Dropout(configs.l1_dropout)
        )

        self.conv_block2_f = nn.Sequential(
            nn.Conv1d(in_channels=configs.l2_input_channels, 
                      out_channels=configs.l2_output_channels, 
                      kernel_size=configs.l2_kernel_size, 
                      stride=configs.l2_stride, 
                      bias=False, 
                      padding=configs.l2_padding),
            nn.BatchNorm1d(configs.l2_output_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=configs.l2_maxpool1d_kernel_size, 
                         stride=configs.l2_maxpool1d_stride, 
                         padding=configs.l2_maxpool1d_padding)
        )

        self.conv_block3_f = nn.Sequential(
            nn.Conv1d(in_channels=configs.l3_input_channels,
                      out_channels=configs.l3_output_channels, 
                      kernel_size=configs.l3_kernel_size, 
                      stride=configs.l3_stride, 
                      bias=False, 
                      padding=configs.l3_padding),
            nn.BatchNorm1d(configs.l3_output_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=configs.l3_maxpool1d_kernel_size, 
                         stride=configs.l3_maxpool1d_stride, 
                         padding=configs.l3_maxpool1d_padding)
        )

        ###################################
        # Cross-Space Linear Projector
        ###################################
        
        # Calculate the size of the output from the convolutional layers (same size if full spectrum is used)
        self.encoder_output_size, _ = self._get_conv_output_size(configs.l1_input_channels, configs.num_time_steps)
        
        self.projector_t = nn.Sequential(
            nn.Linear(in_features=self.encoder_output_size, 
                      out_features=configs.cross_output_channels),
            nn.BatchNorm1d(configs.cross_output_channels),
            nn.ReLU(),
            nn.Linear(configs.cross_output_channels, 128)
        )
        
        self.projector_f = nn.Sequential(
            nn.Linear(in_features=self.encoder_output_size, 
                      out_features=configs.cross_output_channels),
            nn.BatchNorm1d(configs.cross_output_channels),
            nn.ReLU(),
            nn.Linear(configs.cross_output_channels, 128)
        )
        
    def _get_conv_output_size(self, num_channels, num_timesteps):
        """
        Helper function to determine the output size of the convolutional layers.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, num_timesteps)
            output_t = self.conv_block1_t(dummy_input)
            output_t = self.conv_block2_t(output_t)
            output_t = self.conv_block3_t(output_t)
            output_f = self.conv_block1_f(dummy_input)
            output_f = self.conv_block2_f(output_f)
            output_f = self.conv_block3_f(output_f)
            return output_t.numel(), output_f.numel()
    
    def forward(self, x_in_t, x_in_f):
        x_in_t, x_in_f = x_in_t.to(device), x_in_f.to(device)
        
        # Time-based Encoder
        x_t = self.conv_block1_t(x_in_t)
        x_t = self.conv_block2_t(x_t)
        x_t = self.conv_block3_t(x_t)
        h_time = x_t.reshape(x_t.shape[0], -1) # Output of time encoder
        
        z_time = self.projector_t(h_time) # Output of cross_space linear projector for the time encoder
        
        # Frequency-based Encoder
        x_f = self.conv_block1_f(x_in_f)
        x_f = self.conv_block2_f(x_f)
        x_f = self.conv_block3_f(x_f)
        h_freq = x_f.reshape(x_f.shape[0], -1) # Output of freq encoder
        
        z_freq = self.projector_f(h_freq) # Output of cross_space linear projector for the freq encoder

        return h_time, z_time, h_freq, z_freq

    def compute_backbone_combined_loss(self, h_time, h_time_aug, h_freq, h_freq_aug, z_time, z_time_aug, z_freq, z_freq_aug):
        """
        Computes the combined loss for time, frequency, and cross-space consistency in pre-training mode.

        Parameters:
        - h_time: Temporal embeddings of original data.
        - h_time_aug: Temporal embeddings of augmented data.
        - h_freq: Frequency embeddings of original data.
        - h_freq_aug: Frequency embeddings of augmented data.
        - z_time: Projected temporal embeddings of original data.
        - z_time_aug: Projected temporal embeddings of augmented data.
        - z_freq: Projected frequency embeddings of original data.
        - z_freq_aug: Projected frequency embeddings of augmented data.

        Returns:
        - loss_time_encoder: Loss for the time encoder.
        - loss_freq_encoder: Loss for the frequency encoder.
        - loss_consistency: Consistency loss.
        - total_loss: Combined loss value.
        """
        # Time and Frequency loss terms
        loss_time_encoder = self.nt_xent_criterion(h_time, h_time_aug)  # L_t
        loss_freq_encoder = self.nt_xent_criterion(h_freq, h_freq_aug)  # L_f

        # Consistency loss term
        S_T_F = self.nt_xent_criterion(z_time, z_freq)
        S_Taug_F = self.nt_xent_criterion(z_time_aug, z_freq)
        S_T_Faug = self.nt_xent_criterion(z_time, z_freq_aug)
        S_Taug_Faug = self.nt_xent_criterion(z_time_aug, z_freq_aug)

        constant = 10
        loss_consistency = (S_T_F - S_T_Faug + constant) + (S_T_F - S_Taug_F + constant) + (S_T_F - S_Taug_Faug + constant)

        
        # Prints for debugging. Uncomment if running notebook 1
        # print("\n" + "-"*25)
        # print("First Consistency Loss Term:")
        # print(f"(S_T_F - S_T_Faug + constant):\n{S_T_F - S_T_Faug + constant}")
        # print("-"*25)

        # print("Second Consistency Loss Term:")
        # print(f"(S_T_F - S_Taug_F + constant):\n{S_T_F - S_Taug_F + constant}")
        # print("-"*25)

        # print("Third Consistency Loss Term:")
        # print(f"(S_T_F - S_Taug_Faug + constant):\n{S_T_F - S_Taug_Faug + constant}")
        # print("-"*25)

        # print("Total Consistency Loss Term:")
        # print(f"loss_consistency:\n{loss_consistency}")
        # print("-"*25 + "\n")
        
        ## Comment if running notebook 1
        ## Ensure non-negative consistency loss (don't know why this gives me negative loss, but it does. 
        loss_consistency = torch.clamp(loss_consistency, min=0.0)
        
        # Combining all the loss terms
        total_loss = self.lambda_val * (loss_time_encoder + loss_freq_encoder) + (1 - self.lambda_val) * loss_consistency

        return loss_time_encoder, loss_freq_encoder, loss_consistency, total_loss

    def training_step(self, batch, batch_idx):
        time_data, time_aug_data, freq_data, freq_aug_data, labels = batch
        labels = labels.long()
        h_time, z_time, h_freq, z_freq = self(time_data, freq_data)
        h_time_aug, z_time_aug, h_freq_aug, z_freq_aug = self(time_aug_data, freq_aug_data)
        
        loss_time_encoder, loss_freq_encoder, loss_consistency, total_loss = self.compute_backbone_combined_loss(
            h_time, h_time_aug, 
            h_freq, h_freq_aug, 
            z_time, z_time_aug, 
            z_freq, z_freq_aug)
        
        if self.log_metrics:
            self.log('train_loss_total', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_loss_time_enc', loss_time_encoder, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_loss_freq_enc', loss_freq_encoder, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_loss_consist', loss_consistency, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        time_data, time_aug_data, freq_data, freq_aug_data, labels = batch
        labels = labels.long()
        h_time, z_time, h_freq, z_freq = self(time_data, freq_data)
        h_time_aug, z_time_aug, h_freq_aug, z_freq_aug = self(time_aug_data, freq_aug_data)
        
        loss_time_encoder, loss_freq_encoder, loss_consistency, total_loss = self.compute_backbone_combined_loss(
            h_time, h_time_aug, 
            h_freq, h_freq_aug, 
            z_time, z_time_aug, 
            z_freq, z_freq_aug)
        
        if self.log_metrics:
            self.log('val_loss_total', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss_time_enc', loss_time_encoder, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss_freq_enc', loss_freq_encoder, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss_consist', loss_consistency, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['train_loss_total']
        avg_loss_time_enc = self.trainer.callback_metrics['train_loss_time_enc']
        avg_loss_freq_enc = self.trainer.callback_metrics['train_loss_freq_enc']
        avg_loss_consist = self.trainer.callback_metrics['train_loss_consist']
        
        self.train_losses.append(avg_loss.item())
        self.train_loss_time_enc.append(avg_loss_time_enc.item())
        self.train_loss_freq_enc.append(avg_loss_freq_enc.item())
        self.train_loss_consist.append(avg_loss_consist.item())


    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['val_loss_total']
        avg_loss_time_enc = self.trainer.callback_metrics['val_loss_time_enc']
        avg_loss_freq_enc = self.trainer.callback_metrics['val_loss_freq_enc']
        avg_loss_consist = self.trainer.callback_metrics['val_loss_consist']
        
        self.val_loss_consist.append(avg_loss_consist.item())
        self.val_losses.append(avg_loss.item())
        self.val_loss_time_enc.append(avg_loss_time_enc.item())
        self.val_loss_freq_enc.append(avg_loss_freq_enc.item())
        
    def freeze_weights(self):
        """
        Freezes the weights of the model, preventing them from being updated during training.
        """
        for param in self.parameters():
            param.requires_grad = False
        

    def unfreeze_weights(self):
        """
        Unfreezes the weights of the model, allowing them to be updated during training.
        """
        for param in self.parameters():
            param.requires_grad = True
    
    def check_if_frozen(self):
        for name, param in self.named_parameters():
            print(f"Parameter: {name}, Requires_grad: {param.requires_grad}")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.optimizer_config.lr, 
                                     betas=(self.optimizer_config.beta1, self.optimizer_config.beta2),
                                     weight_decay=self.optimizer_config.weight_decay)
        return optimizer


#########################################
# TFC Projector Head (Or Prediction Head)
#########################################

class TFC_Projector_Head(L.LightningModule):
    def __init__(self,  global_config: GlobalConfigFile):
        super(TFC_Projector_Head, self).__init__()
        self.logits = nn.Linear(2 * 128, 64)  # We multiply by 2 since we concatenate the embeddings of the two encoders
        self.logits_simple = nn.Linear(64, 6)
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer_config = global_config.optimizer_config
        
    def forward(self, emb):
        # 2-layer MLP
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred

    def compute_projector_head_loss(self, predictions, labels):
        """
        Computes the CrossEntropyLoss for the predictions and labels.

        Parameters:
        - predictions: Predicted class probabilities from the model.
        - labels: Ground truth labels.

        Returns:
        - loss: Computed CrossEntropyLoss.
        """
        return self.criterion(predictions, labels)

    def training_step(self, batch, batch_idx):
        _, _, _, _, labels, h_time, h_freq = batch
        labels = labels.long()
        h_combined = torch.cat((h_time, h_freq), dim=1)
        predictions = self(h_combined)
        loss = self.compute_projector_head_loss(predictions, labels)
        
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, _, _, labels, h_time, h_freq = batch
        labels = labels.long()
        h_combined = torch.cat((h_time, h_freq), dim=1)
        predictions = self(h_combined)
        loss = self.compute_projector_head_loss(predictions, labels)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.optimizer_config.lr, 
                                     betas=(self.optimizer_config.beta1, self.optimizer_config.beta2),
                                     weight_decay=self.optimizer_config.weight_decay)
        return optimizer



##############################################
# Combining Backbone and Projector Head
##############################################

class TFC_Combined_Model(L.LightningModule):
    def __init__(self, global_config: GlobalConfigFile, backbone=None, projector_head=None):
        super(TFC_Combined_Model, self).__init__()
        
        # If we want to use an already trained backbone
        if backbone:
            self.backbone = backbone
        else:
            self.backbone = TFC_Backbone(global_config, log_metrics=False)
        
        # If we want to use another kind of projector head
        if projector_head:
            self.projection_head = projector_head
        else:
            self.projection_head = TFC_Projector_Head(global_config)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=6)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=6)
        self.train_recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=6)
        self.val_recall = torchmetrics.Recall(task='multiclass', average='macro', num_classes=6)
        self.train_f1 = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=6)
        self.val_f1 = torchmetrics.F1Score(task='multiclass', average='macro', num_classes=6)

        # Initialize metric logs
        self.train_loss_log = []
        self.val_loss_log = []
        self.train_accuracy_log = []
        self.val_accuracy_log = []
        self.train_recall_log = []
        self.val_recall_log = []
        self.train_f1_log = []
        self.val_f1_log = []
        self.train_loss_backbone = []
        self.val_loss_backbone = []
        self.train_loss_proj_head = []
        self.val_loss_proj_head = []
        self.train_loss_time_enc = []
        self.val_loss_time_enc = []
        self.train_loss_freq_enc = []
        self.val_loss_freq_enc = []
        self.train_loss_consist = []
        self.val_loss_consist = []
        self.initial_validation = True

        self.optimizer_config = global_config.optimizer_config

    def forward(self, time_data, freq_data):
        h_time, z_time, h_freq, z_freq = self.backbone(time_data, freq_data)
        h_combined = torch.cat((z_time, z_freq), dim=1)
        predictions = self.projection_head(h_combined)
        return predictions, (h_time, z_time, h_freq, z_freq)

    def compute_combined_loss(self, time_data, time_aug_data, freq_data, freq_aug_data, labels):
        # Forward pass through the backbone
        h_time, z_time, h_freq, z_freq = self.backbone(time_data, freq_data)
        h_time_aug, z_time_aug, h_freq_aug, z_freq_aug = self.backbone(time_aug_data, freq_aug_data)

        # Compute backbone loss
        loss_time_encoder, loss_freq_encoder, loss_consistency, backbone_loss = self.backbone.compute_backbone_combined_loss(
            h_time, h_time_aug,
            h_freq, h_freq_aug,
            z_time, z_time_aug, 
            z_freq, z_freq_aug)

        # Forward pass through the projector head
        h_combined = torch.cat((z_time, z_freq), dim=1)
        predictions = self.projection_head(h_combined)

        # Compute projection head loss
        proj_head_loss = self.criterion(predictions, labels)

        # Combine the losses
        total_loss = backbone_loss + proj_head_loss
        return total_loss, backbone_loss, proj_head_loss, loss_time_encoder, loss_freq_encoder, loss_consistency

    def training_step(self, batch, batch_idx):
        time_data, time_aug_data, freq_data, freq_aug_data, labels = batch
        labels = labels.long()
        total_loss, backbone_loss, proj_head_loss, loss_time_encoder, loss_freq_encoder, loss_consistency = self.compute_combined_loss(
            time_data, time_aug_data, 
            freq_data, freq_aug_data, 
            labels)

        # Log metrics
        self.log('train_loss_total', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_backbone', backbone_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_loss_proj_head', proj_head_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_loss_time_enc', loss_time_encoder, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_loss_freq_enc', loss_freq_encoder, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_loss_consist', loss_consistency, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        predictions, _ = self(time_data, freq_data)
        self.train_accuracy(predictions, labels)
        self.train_recall(predictions, labels)
        self.train_f1(predictions, labels)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        time_data, time_aug_data, freq_data, freq_aug_data, labels = batch
        labels = labels.long()
        total_loss, backbone_loss, proj_head_loss, loss_time_encoder, loss_freq_encoder, loss_consistency = self.compute_combined_loss(
            time_data, time_aug_data, 
            freq_data, freq_aug_data, 
            labels)

        # Log metrics
        self.log('val_loss_total', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_backbone', backbone_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss_proj_head', proj_head_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss_time_enc', loss_time_encoder, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss_freq_enc', loss_freq_encoder, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_loss_consist', loss_consistency, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        predictions, _ = self(time_data, freq_data)
        self.val_accuracy(predictions, labels)
        self.val_recall(predictions, labels)
        self.val_f1(predictions, labels)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['train_loss_total']
        avg_backbone_loss = self.trainer.callback_metrics['train_loss_backbone']
        avg_proj_head_loss = self.trainer.callback_metrics['train_loss_proj_head']
        avg_time_enc_loss = self.trainer.callback_metrics.get('train_loss_time_enc')
        avg_freq_enc_loss = self.trainer.callback_metrics.get('train_loss_freq_enc')
        avg_consist_loss = self.trainer.callback_metrics.get('train_loss_consist')
        avg_acc = self.trainer.callback_metrics['train_accuracy']
        avg_recall = self.trainer.callback_metrics['train_recall']
        avg_f1 = self.trainer.callback_metrics['train_f1']

        self.train_loss_log.append(avg_loss.item())
        self.train_loss_backbone.append(avg_backbone_loss.item())
        self.train_loss_proj_head.append(avg_proj_head_loss.item())
        
        self.train_loss_time_enc.append(avg_time_enc_loss.item())
        self.train_loss_freq_enc.append(avg_freq_enc_loss.item())
        self.train_loss_consist.append(avg_consist_loss.item())
        
        self.train_accuracy_log.append(avg_acc.item())
        self.train_recall_log.append(avg_recall.item())
        self.train_f1_log.append(avg_f1.item())

        # Reset metrics
        self.train_accuracy.reset()
        self.train_recall.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        # if self.initial_validation:
        #     self.initial_validation = False
        #     return
        
        avg_loss = self.trainer.callback_metrics['val_loss_total']
        avg_backbone_loss = self.trainer.callback_metrics['val_loss_backbone']
        avg_proj_head_loss = self.trainer.callback_metrics['val_loss_proj_head']
        avg_time_enc_loss = self.trainer.callback_metrics.get('val_loss_time_enc')
        avg_freq_enc_loss = self.trainer.callback_metrics.get('val_loss_freq_enc')
        avg_consist_loss = self.trainer.callback_metrics.get('val_loss_consist')
        avg_acc = self.trainer.callback_metrics['val_accuracy']
        avg_recall = self.trainer.callback_metrics['val_recall']
        avg_f1 = self.trainer.callback_metrics['val_f1']

        self.val_loss_log.append(avg_loss.item())
        self.val_loss_backbone.append(avg_backbone_loss.item())
        self.val_loss_proj_head.append(avg_proj_head_loss.item())
        
        self.val_loss_time_enc.append(avg_time_enc_loss.item())
        self.val_loss_freq_enc.append(avg_freq_enc_loss.item())
        self.val_loss_consist.append(avg_consist_loss.item())
        
        self.val_accuracy_log.append(avg_acc.item())
        self.val_recall_log.append(avg_recall.item())
        self.val_f1_log.append(avg_f1.item())
        
        # Reset metrics
        self.val_accuracy.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.optimizer_config.lr, 
                                     betas=(self.optimizer_config.beta1, self.optimizer_config.beta2),
                                     weight_decay=self.optimizer_config.weight_decay)
        return optimizer

    def freeze_backbone(self):
        """
        Freezes the weights of the model, preventing them from being updated during training.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        # self.check_if_frozen()

    def unfreeze_backbone(self):
        """
        Unfreezes the weights of the model, allowing them to be updated during training.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        # self.check_if_frozen()

    def check_if_frozen(self):
        """Prints the requires_grad status of the backbone and projection head parameters."""
        for name, param in self.backbone.named_parameters():
            print(f"Backbone Parameter: {name}, Requires_grad: {param.requires_grad}")
        for name, param in self.projection_head.named_parameters():
            print(f"Projection Head Parameter: {name}, Requires_grad: {param.requires_grad}")

######################
# Loss function
#######################
class NTXentLoss(nn.Module):
    def __init__(self, global_config: GlobalConfigFile):
        super(NTXentLoss, self).__init__()
        self.config = global_config.loss_config
        self.batch_size = self.config.batch_size
        self.temperature = self.config.temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(self.config.use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.use_poly_loss = self.config.use_poly_loss

    def _get_similarity_function(self, use_cosine_similarity):
        """
        Chooses the similarity function (cosine similarity or dot product).

        Args:
        - use_cosine_similarity (bool): If True, use cosine similarity. Otherwise, use dot product.

        Returns:
        - function: The selected similarity function.
        """
        if use_cosine_similarity:
            self._cosine_similarity = nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity
        else:
            return self._dot_similarity

    def _get_correlated_mask(self):
        """
        Generates a mask to filter out the diagonal elements (self-similarity) and the correlated samples from the same representation.

        Returns:
        - mask (torch.Tensor): A mask tensor to filter out correlated samples.
        """
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_similarity(x, y):
        """
        Computes the dot product similarity between x and y.

        Args:
        - x (torch.Tensor): Input tensor of shape (N, C).
        - y (torch.Tensor): Input tensor of shape (M, C).

        Returns:
        - v (torch.Tensor): Similarity matrix of shape (N, M).
        """
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_similarity(self, x, y):
        """
        Computes the cosine similarity between x and y.

        Args:
        - x (torch.Tensor): Input tensor of shape (N, C).
        - y (torch.Tensor): Input tensor of shape (M, C).

        Returns:
        - v (torch.Tensor): Similarity matrix of shape (N, M).
        """
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        """
        Computes the NT-Xent loss.

        Args:
        - zis (torch.Tensor): Batch of transformed representations for one augmentation (shape: [batch_size, dim]).
        - zjs (torch.Tensor): Batch of transformed representations for another augmentation (shape: [batch_size, dim]).

        Returns:
        - loss (torch.Tensor): The computed NT-Xent loss.
        """
        if zis.shape[0] != self.batch_size or zjs.shape[0] != self.batch_size:
            print(f"Batch size mismatch. Expected: {self.batch_size}, Got: {zis.shape[0]}, {zjs.shape[0]}")
            return None

        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        # Filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        # Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0.
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        if self.use_poly_loss:
            onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1), torch.zeros(2 * self.batch_size, negatives.shape[-1])), dim=-1).to(self.device).long()
            pt = torch.mean(onehot_label * torch.nn.functional.softmax(logits, dim=-1))

            epsilon = self.batch_size
            loss = CE / (2 * self.batch_size) + epsilon * (1 / self.batch_size - pt)
        else:
            loss = CE / (2 * self.batch_size)
        
        return loss




###############################
# Callbacks 
###############################
    
class TFCCallbacks:
    """
    A class to set up and manage callbacks for a PyTorch Lightning training process, 
    including early stopping, model checkpointing, and TensorBoard logging.

    Attributes:
        monitor (str): Metric to monitor for early stopping and checkpointing.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        verbose (bool): If True, prints a message for each validation improvement.
        log_dir (str): Directory to save logs and checkpoints.
        experiment_name (str): Name of the experiment.
        version (int): Version of the experiment.
        SSL_technique_prefix (str): Prefix for checkpoint filenames.
        experiment_path (str): Path to the experiment directory.
        checkpoint_path (str): Path to the checkpoint directory.
        tensorboard_path (str): Path to the TensorBoard log directory.
    """
    
    def __init__(self, monitor='val_loss_total', min_delta=0.001, patience=5, verbose=True,
                 log_dir='lightning_logs', experiment_name='tfc_experiment', version=0, SSL_technique_prefix='TF_C'):
        """
        Initializes the TFCCallbacks with the given parameters and ensures the directory structure exists.

        Args:
            monitor (str): Metric to monitor for early stopping and checkpointing.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            verbose (bool): If True, prints a message for each validation improvement.
            log_dir (str): Directory to save logs and checkpoints.
            experiment_name (str): Name of the experiment.
            version (int): Version of the experiment.
            SSL_technique_prefix (str): Prefix for checkpoint filenames.
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.version = version
        self.SSL_technique_prefix = SSL_technique_prefix

        # Define paths for the experiment
        self.experiment_path = f"{self.log_dir}/{self.experiment_name}/{self.version}"
        self.checkpoint_path = os.path.join(self.experiment_path, 'checkpoints')
        self.tensorboard_path = os.path.join(self.experiment_path, 'tensorboard')

        # Ensure directories exist
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.tensorboard_path, exist_ok=True)

    def get_callbacks(self):
        """
        Creates and returns the early stopping and model checkpoint callbacks.

        Returns:
            list: A list containing the early stopping and model checkpoint callbacks.
        """
        early_stopping = EarlyStopping(
            monitor=self.monitor,
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=self.verbose
        )
        
        checkpoint = ModelCheckpoint(
            monitor=self.monitor,
            save_top_k=1,
            mode='min',
            dirpath=self.checkpoint_path,
            filename=f'{self.SSL_technique_prefix}-model' + '-{epoch}-{train_loss_total:.2f}',
            save_weights_only=True
        )
        
        return [early_stopping, checkpoint]

    def get_logger(self):
        """
        Creates and returns the TensorBoard logger.

        Returns:
            TensorBoardLogger: The TensorBoard logger.
        """
        return TensorBoardLogger(
            save_dir=self.tensorboard_path,
            name='',
            version=''
        )