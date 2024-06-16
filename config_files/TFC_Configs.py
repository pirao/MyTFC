import numpy as np
import pandas as pd
import torch
from torch import nn

####################################################################################################
####################################################################################################
# Configuration settings for time and frequency augmentations.
####################################################################################################
####################################################################################################


class TimeAugmentConfigSettings:
    def __init__(self, mask_keepratio=0.9, jitter_sigma=0.8, scaling_sigma=1.1, permut_max_segments=5):
        """
        Configuration settings for time domain augmentations.

        Parameters:
        - mask_keepratio: Ratio of values to keep during masking.
        - jitter_sigma: Standard deviation for jittering.
        - scaling_sigma: Standard deviation for scaling.
        - permut_max_segments: Maximum number of segments for permutation.
        """
        self.mask_keepratio = mask_keepratio
        self.jitter_sigma = jitter_sigma
        self.scaling_sigma = scaling_sigma
        self.permut_max_segments = permut_max_segments
        


    def display_params(self):
        """
        Displays the configuration parameters.
        """
        print("TimeAugmentConfigSettings:")
        print(f"  mask_keepratio: {self.mask_keepratio}")
        print(f"  jitter_sigma: {self.jitter_sigma}")
        print(f"  scaling_sigma: {self.scaling_sigma}")
        print(f"  permut_max_segments: {self.permut_max_segments}")        
        

class FreqAugmentConfigSettings:
    def __init__(self, remove_freq_ratio=0.2, add_freq_ratio=0.2, add_freq_constant=0.2):
        """
        Configuration settings for frequency domain augmentations.

        Parameters:
        - remove_freq_ratio: Ratio of frequencies to remove.
        - add_freq_ratio: Ratio of frequencies to add.
        - add_freq_constant: Constant for frequency addition.
        """
        self.remove_freq_ratio = remove_freq_ratio
        self.add_freq_ratio = add_freq_ratio
        self.add_freq_constant = add_freq_constant

    def display_params(self):
        """
        Displays the configuration parameters.
        """
        print("FreqAugmentConfigSettings:")
        print(f"  remove_freq_ratio: {self.remove_freq_ratio}")
        print(f"  add_freq_ratio: {self.add_freq_ratio}")
        print(f"  add_freq_constant: {self.add_freq_constant}")

####################################################################################################
####################################################################################################
# Configuration for the encoder and decoder models.
####################################################################################################
####################################################################################################

class EncoderConfigSettings:
    def __init__(self, input_shape=6, 
                 l1_output_channels=32, l1_kernel_size=8, l1_stride=8,
                 l1_maxpool1d_kernel_size=2, l1_maxpool1d_stride=2, l1_maxpool1d_padding=1, l1_dropout=0,
                 l2_output_channels=64, l2_kernel_size=8, l2_stride=1,
                 l2_maxpool1d_kernel_size=2, l2_maxpool1d_stride=2, l2_maxpool1d_padding=1,
                 l3_output_channels=6, l3_kernel_size=8, l3_stride=1,
                 l3_maxpool1d_kernel_size=2, l3_maxpool1d_stride=2, l3_maxpool1d_padding=1,
                 cross_output_channels=256, num_time_steps=60):
        """
        Configuration settings for the encoders.

        Parameters:
        - input_shape: Number of input channels (default is 6).
        - l1_output_channels: Number of output channels for the 1st convolutional layer (default is 32).
        - l1_kernel_size: Kernel size for the 1st convolutional layer (default is 16).
        - l1_stride: Stride for the 1st convolutional layer (default is 2).
        - l1_maxpool1d_kernel_size: Kernel size for the 1st max-pooling layer (default is 2).
        - l1_maxpool1d_stride: Stride for the 1st max-pooling layer (default is 2).
        - l1_maxpool1d_padding: Padding for the 1st max-pooling layer (default is 1).
        - l1_dropout: Dropout rate for the 1st layer (default is 0).
        
        - l2_output_channels: Number of output channels for the 2nd convolutional layer (default is 64).
        - l2_kernel_size: Kernel size for the 2nd convolutional layer (default is 8).
        - l2_stride: Stride for the 2nd convolutional layer (default is 1).
        - l2_maxpool1d_kernel_size: Kernel size for the 2nd max-pooling layer (default is 2).
        - l2_maxpool1d_stride: Stride for the 2nd max-pooling layer (default is 2).
        - l2_maxpool1d_padding: Padding for the 2nd max-pooling layer (default is 1).
        
        - l3_output_channels: Number of output channels for the 3rd convolutional layer (default is 6).
        - l3_kernel_size: Kernel size for the 3rd convolutional layer (default is 8).
        - l3_stride: Stride for the 3rd convolutional layer (default is 1).
        - l3_maxpool1d_kernel_size: Kernel size for the 3rd max-pooling layer (default is 2).
        - l3_maxpool1d_stride: Stride for the 3rd max-pooling layer (default is 2).
        - l3_maxpool1d_padding: Padding for the 3rd max-pooling layer (default is 1).
        
        - cross_output_channels: Number of output channels for the final fully connected layers.
        
        - num_time_steps: Length of the input time series (default is 60).
        """

        # 1st layer
        self.l1_input_channels = input_shape
        self.l1_output_channels = l1_output_channels
        self.l1_kernel_size = l1_kernel_size
        self.l1_stride = l1_stride
        self.l1_padding = (self.l1_kernel_size) // 2  # 'valid', 'same' or tuple
        self.l1_maxpool1d_kernel_size = l1_maxpool1d_kernel_size
        self.l1_maxpool1d_stride = l1_maxpool1d_stride
        self.l1_maxpool1d_padding = l1_maxpool1d_padding
        self.l1_dropout = l1_dropout

        # 2nd layer
        self.l2_input_channels = l1_output_channels  # The output of layer 1 is the input of layer 2
        self.l2_output_channels = l2_output_channels
        self.l2_kernel_size = l2_kernel_size
        self.l2_stride = l2_stride
        self.l2_padding = (self.l2_kernel_size) // 2  
        self.l2_maxpool1d_kernel_size = l2_maxpool1d_kernel_size
        self.l2_maxpool1d_stride = l2_maxpool1d_stride
        self.l2_maxpool1d_padding = l2_maxpool1d_padding

        # 3rd and final layer
        self.l3_input_channels = l2_output_channels  # The output of layer 2 is the input of layer 3
        self.l3_output_channels = l3_output_channels
        self.l3_kernel_size = l3_kernel_size
        self.l3_stride = l3_stride
        self.l3_padding = (self.l3_kernel_size) // 2  
        self.l3_maxpool1d_kernel_size = l3_maxpool1d_kernel_size
        self.l3_maxpool1d_stride = l3_maxpool1d_stride
        self.l3_maxpool1d_padding = l3_maxpool1d_padding
        
        self.cross_output_channels = cross_output_channels
        
        self.num_time_steps = num_time_steps
        
    def display_params(self):
        """
        Displays the configuration parameters.
        """
        print("EncoderConfigSettings:")
        for key, value in self.__dict__.items():
            print(f"  {key}: {value}")
            
                     
            
####################################################################################################
####################################################################################################
# Global configuration settings for the TFC model.
####################################################################################################
####################################################################################################


class LossConfigSettings:
    def __init__(self, temperature=0.5, use_cosine_similarity=False, lambda_val=0.2, batch_size=64, use_poly_loss=True):
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.lambda_val = lambda_val
        self.batch_size = batch_size
        self.use_poly_loss = use_poly_loss
        
    def display_params(self):
        print("LossConfigSettings:")
        for key, value in self.__dict__.items():
            print(f"  {key}: {value}")  


class OptimizerConfigSettings:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.99,weight_decay=3e-4):
        """
        Configuration settings the optimizer being used
.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        
    def display_params(self):
        """
        Displays the configuration parameters.
        """
        print("OptimizerConfigSettings:")
        for key, value in self.__dict__.items():
            print(f"  {key}: {value}") 



class GlobalConfigFile:
    def __init__(self, batch_size=64):
        """
        Global configuration file that includes:
        1. Time and frequency augmentation hyperparameters.
        2. Encoder architectures hyperparameters
        3. Loss hyperparameters 
        4. Optimizer hyperparameters
        
        Each configuration files contains the hyperparameters used for the specific configuration
        
        """
        self.batch_size = batch_size
        
        # Augmentation settings
        self.time_aug_config = TimeAugmentConfigSettings()
        self.freq_aug_config = FreqAugmentConfigSettings()
        
        # Loss function settings
        self.loss_config = LossConfigSettings(batch_size=self.batch_size)
        
        # Optimizer hyperparameters
        self.optimizer_config = OptimizerConfigSettings()
        
        # Encoder settings
        self.enc_config = EncoderConfigSettings()