import numpy as np

import sys
sys.path.append('../TFC_Configs/')

from config_files.TFC_Configs import *
from transforms.tfc_augmentations import *


########################################
########################################
# Selecting the augmentation function
########################################
########################################

def one_hot_encoding(X, num_classes):
    """
    Converts a tensor or list of class labels to one-hot encoding.

    Parameters:
    - X: Tensor or list of class labels.
    - num_classes: Total number of classes.

    Returns:
    - One-hot encoded numpy array.

    Example:
    >>> one_hot_encoding(torch.tensor([0, 1, 2, 1]), 4)
    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 1., 0., 0.]])
    """
    if isinstance(X, torch.Tensor):
        # Convert the class labels to integers if they are not already
        X = X.long()
        one_hot_encoded = torch.nn.functional.one_hot(X, num_classes=num_classes).float()
        return one_hot_encoded.numpy()
    else:
        # Convert the class labels to integers if they are not already
        X = [int(x) for x in X]
        # Use numpy's eye function to create one-hot encoding
        return np.eye(num_classes)[X]



########################################
########################################
# Data Augmentation - Time series
#########################################
#########################################


def jitter(x, sigma=0.6):
    """
    Applies jitter augmentation to the input data.
    
    Parameters:
    - x: Input data.
    - sigma: Standard deviation of the jitter noise (default is 0.6).
    
    Returns:
    - Jittered data.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be a torch tensor, but got {type(x)}.")
    return x + torch.randn_like(x) * sigma

def scaling(x, sigma=1.1):
    """
    Applies scaling augmentation to the input data. The scaling factor is drawn from a normal distribution
    and multiplies each value in the time series.

    Parameters:
    - x: Input data (torch tensor) with shape (num_sensors, num_timesteps).
    - sigma: Standard deviation of the scaling factor (default is 1.1).

    Returns:
    - Scaled data (torch tensor) with the same shape as the input.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be a torch tensor, but got {type(x)}.")
    
    scaling_factor = torch.normal(mean=2.0, std=sigma, size=(1, x.shape[1]), device=x.device)
    return x * scaling_factor

def permutation(x, max_segments=5, seg_mode="random"):
    """
    Applies permutation augmentation to the input data. The time series is split into segments and the segments are randomly permuted.
    
    Parameters:
    - x: Input data tensor (time series).
    - max_segments: Maximum number of segments to split the time series into (default is 5).
    - seg_mode: Mode of segmentation. Can be "random" for random splits or "equal" for equal splits (default is "random").
    
    Returns:
    - permuted_data: Permuted time series data.
    - affected_indices: Indices of the original time series that were permuted.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be a torch tensor, but got {type(x)}.")
    
    orig_steps = torch.arange(x.size(0))
    num_segs = min(max_segments, x.size(0))

    if seg_mode == "random":
        split_points = torch.randint(1, x.size(0), (num_segs - 1,))
        split_points, _ = torch.sort(split_points)
    else:
        split_points = torch.arange(1, x.size(0), x.size(0) // num_segs)[:num_segs - 1]

    segments = torch.split(orig_steps, torch.diff(torch.cat((torch.tensor([0]), split_points, torch.tensor([x.size(0)])))).tolist())
    permuted_segments = [segments[i] for i in torch.randperm(len(segments))]
    permuted_indices = torch.cat(permuted_segments)

    affected_indices = (orig_steps != permuted_indices).nonzero().squeeze()
    return x[permuted_indices], affected_indices

def masking(x, keepratio=0.9, mask='binomial'):
    """
    Applies masking to the input data. The selected values are set to zero.
    
    Parameters:
    - x: Input data array (row vector).
    - keepratio: The ratio of values to keep (default is 0.9).
    - mask: The type of mask to use (default is 'binomial').
    
    Returns:
    - Masked data array and affected indices.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input must be a torch tensor, but got {type(x)}.")
    x2 = x.clone()
    nan_mask = ~torch.isnan(x)
    x2[~nan_mask] = 0
    
    if mask == 'binomial':
        mask_id = torch.bernoulli(torch.full_like(x, keepratio)).bool()
    else:
        raise ValueError(f"Mask type '{mask}' not supported.")
    
    x2[~mask_id] = 0
    affected_indices = torch.where(~mask_id)[0]
    return x2, affected_indices

    
########################################
########################################
# Data Augmentation - Frequency domain
#########################################
#########################################

def remove_frequency(amplitude_spectrum, remove_ratio=0.0):
    """
    Removes a percentage of amplitude components from the input amplitude spectrum.

    Parameters:
    - amplitude_spectrum: Real-valued amplitude spectrum (half spectrum).
    - remove_ratio: Ratio of components to remove.

    Returns:
    - Modified amplitude spectrum.
    """
    if not isinstance(amplitude_spectrum, torch.Tensor):
        raise TypeError(f"Input must be a torch tensor, but got {type(amplitude_spectrum)}.")
    if torch.is_complex(amplitude_spectrum):
        raise ValueError(f"Input amplitude spectrum must be real-valued, but got complex values.")
    mask = torch.rand_like(amplitude_spectrum) > remove_ratio
    return amplitude_spectrum * mask

def add_frequency(amplitude_spectrum, add_ratio=0.0, const=0.2):
    """
    Adds random frequency components to the input signal. If the random number is greater than (1 - perturb_ratio), we get a list of
    the frequency components. From this list, we get the maximum amplitude and use it to define a uniform
    distribution. The scaling factor alpha is sampled from this distribution and multiplied by the mask. These results in a list
    of 0 and scaled values. We add this list to the input signal.
    
    Larger perturb_ratio values will result in more frequency components being added.
    
    Parameters:
    - x: Input signal (frequency domain).
    - perturb_ratio: Ratio of frequency components to add. 1.0 means we perturb all frequency components, 0.0 means we perturb nothing.
    
    Returns:
    - Signal with added frequency components.
    """
    if not isinstance(amplitude_spectrum, torch.Tensor):
        raise TypeError(f"Input must be a torch tensor, but got {type(amplitude_spectrum)}.")
    if amplitude_spectrum.is_complex():
        raise ValueError(f"Input amplitude spectrum must be real-valued, but got complex values.")
    
    mask = torch.rand_like(amplitude_spectrum) > add_ratio
    max_amplitude = torch.max(amplitude_spectrum)
    alpha = torch.rand_like(amplitude_spectrum) * max_amplitude * const
    return amplitude_spectrum + mask * alpha

######################################
# TF_C Augmentation Class
######################################

class TFC_transforms:
    """
    Applies time and frequency domain augmentations to the input data based on the given configuration.

    Attributes:
    - global_config_file: Configuration file containing augmentation parameters.
    - verbose: Boolean indicating whether to print detailed augmentation information.

    Methods:
    - transform_time(sample): Applies time domain augmentations to the input time series data.
    - transform_freq(sample): Applies frequency domain augmentations to the input half spectrum data.
    - __call__(x, x_freq): Applies both time and frequency domain augmentations to the input data.
    """
    def __init__(self, global_config_file, verbose=False):
        self.global_config_file = global_config_file
        self.verbose = verbose

    def transform_time(self, time_sample):
        """
        Applies a bank of time domain augmentations to the input time series and randomly selects one as the positive sample (1),
        while the others are the negative samples (0).

        Parameters:
        - time_sample: Input time series data (torch tensor) with shape (num_sensors, num_timesteps).
                       The tensor should represent a single sample of time series data where:
                       - num_sensors: Number of sensors (channels).
                       - num_timesteps: Number of timesteps in each time series.

        Returns:
        - Augmented time series data (torch tensor) with the same shape as the input.
        """
        if not isinstance(time_sample, torch.Tensor):
            raise TypeError(f"Input time_sample must be a torch tensor, but got {type(time_sample)}.")
        if time_sample.dim() != 2:
            raise ValueError(f"Input time_sample must have 2 dimensions (num_sensors, num_timesteps), but got {time_sample.shape}.")
                             
        aug_1 = jitter(time_sample, self.global_config_file.time_aug_config.jitter_sigma)
        aug_2 = scaling(time_sample, self.global_config_file.time_aug_config.scaling_sigma)
        aug_3, _ = permutation(time_sample, self.global_config_file.time_aug_config.permut_max_segments)
        aug_4, _ = masking(time_sample, self.global_config_file.time_aug_config.mask_keepratio)

        li = torch.randint(0, 4, (1,), device=time_sample.device)
        li_onehot = torch.nn.functional.one_hot(li, 4).float()

        aug_1 = aug_1 * li_onehot[:, 0][:, None]
        aug_2 = aug_2 * li_onehot[:, 1][:, None]
        aug_3 = aug_3 * li_onehot[:, 2][:, None]
        aug_4 = aug_4 * li_onehot[:, 3][:, None]

        aug_T = aug_1 + aug_2 + aug_3 + aug_4

        if not (aug_1.shape == aug_2.shape == aug_3.shape == aug_4.shape):
            raise ValueError(f"All augmentations must have the same shape. Got shapes: aug_1 {aug_1.shape}, aug_2 {aug_2.shape}, aug_3 {aug_3.shape}, aug_4 {aug_4.shape}")
        
        if self.verbose:
            for idx, aug_type in enumerate(li):
                aug_name = ['Jitter', 'Scaling', 'Permutation', 'Masking'][aug_type]
                print(f"Time domain: Row {idx} was augmented with {aug_name}")

        if len(aug_T.shape) != 2:
            raise ValueError(f"Input time_sample must have 2 dimensions (num_sensors, num_timesteps), but got {aug_T.shape}.")
        
        
        return aug_T

    def transform_freq(self, freq_sample):
        """
        Applies a bank of frequency domain augmentations to the input half spectrum and randomly selects one as the positive sample (1),
        while the others are the negative samples (0).

        Parameters:
        - freq_sample: Input half spectrum data (torch tensor) with shape (num_sensors, num_freq_bins).
                       The tensor should represent a single sample of frequency domain data where:
                       - num_sensors: Number of sensors (channels).
                       - num_freq_bins: Number of frequency bins in the half spectrum.

        Returns:
        - Augmented half spectrum data (torch tensor) with the same shape as the input.
        """
        if not isinstance(freq_sample, torch.Tensor):
            raise TypeError(f"Input freq_sample must be a torch tensor, but got {type(freq_sample)}.")
        if freq_sample.is_complex():
            raise ValueError("Input half spectrum must be real-valued.")

        aug_1 = remove_frequency(freq_sample, self.global_config_file.freq_aug_config.remove_freq_ratio)
        aug_2 = add_frequency(freq_sample, self.global_config_file.freq_aug_config.add_freq_ratio, self.global_config_file.freq_aug_config.add_freq_constant)

        li = torch.randint(0, 2, (1,), device=freq_sample.device)
        li_onehot = torch.nn.functional.one_hot(li, 2).float()

        aug_1 = aug_1 * li_onehot[:, 0][:, None]
        aug_2 = aug_2 * li_onehot[:, 1][:, None]
        aug_F = aug_1 + aug_2

        if not (aug_1.shape == aug_2.shape == aug_F.shape):
            raise ValueError(f"All augmentations must have the same shape. Got shapes: aug_1 {aug_1.shape}, aug_2 {aug_2.shape}, aug_3 {aug_F.shape}")
        
        if self.verbose:
            for idx, aug_type in enumerate(li):
                aug_name = ['Remove Frequency', 'Add Frequency'][aug_type]
                print(f"Frequency domain: Row {idx} was augmented with {aug_name}")

        return aug_F

    def __call__(self, x, x_freq):
        """
        Applies both time and frequency domain augmentations to the input data.

        Parameters:
        - x: Input time series data (torch tensor, numpy array, or pandas DataFrame) with shape (num_sensors, num_timesteps).
        - x_freq: Input half spectrum data (torch tensor) with shape (num_sensors, num_freq_bins).

        Returns:
        - Tuple containing the original tensor, augmented time series tensor, and augmented half spectrum tensor.
        """
        
        if isinstance(x, pd.DataFrame):
            x_tensor = torch.tensor(x.values, dtype=torch.float32)
        elif isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x

        if isinstance(x_freq, pd.DataFrame):
            x_freq_tensor = torch.tensor(x_freq.values, dtype=torch.float32)
        elif isinstance(x_freq, np.ndarray):
            x_freq_tensor = torch.tensor(x_freq, dtype=torch.float32)
        else:
            x_freq_tensor = x_freq
        
        if not torch.is_tensor(x_tensor) or not torch.is_tensor(x_freq_tensor):
            raise TypeError(f"Both inputs x and x_freq must be tensors. Got types {type(x_tensor)} and {type(x_freq_tensor)}.")

        y1 = self.transform_time(x_tensor)
        y2 = self.transform_freq(x_freq_tensor)
            
        return x_tensor, y1, y2