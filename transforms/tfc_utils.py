import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
from scipy.fft import fft, fftfreq, ifft 

import sys
sys.path.append('../')
from transforms.tfc_augmentations import *

################################
# Random seed
################################

def set_seed(seed):
    """
    Set the random seed for reproducibility across multiple runs.

    Args:
        seed (int): The seed value to set for random number generation.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



#######################
# Sample selection
##################

def sample_unique_classes(df, target):
    """
    Filters the input dataframe and target to obtain a sample of each unique class and its corresponding data.

    Args:
        df (torch.Tensor): The input data tensor with shape (batch_size, number_of_sensors, number_of_timesteps).
        target (torch.Tensor): The target tensor with shape (batch_size).

    Returns:
        torch.Tensor: A tensor containing one sample for each unique class with shape 
                      (number_of_unique_classes, number_of_sensors, number_of_timesteps).
        torch.Tensor: A tensor containing the unique classes.
    
    Example:
        df, target = first_batch
        filtered_df, unique_classes = sample_unique_classes(df, target)
        print("Filtered Dataframe Shape:", filtered_df.shape)
        print("Unique Classes:", unique_classes)
    
    Notes:
        - The function assumes that the input tensors `df` and `target` have matching first dimensions (batch_size).
        - The function selects the first occurrence of each unique class in the target tensor.
    """
    unique_classes = torch.unique(target)
    samples = []

    for cls in unique_classes:
        class_indices = (target == cls).nonzero(as_tuple=True)[0]
        sample_index = class_indices[0]
        
        # Append the selected sample to the list
        samples.append(df[sample_index].unsqueeze(0))

    # Concatenate the collected samples along the first dimension (batch_size)
    filtered_df = torch.cat(samples, dim=0)
    
    return filtered_df, unique_classes


################################
################################
# FFT computation
################################
################################

def compute_normalized_frequency_domain(x, dt):
    """
    Computes the normalized frequency domain representation.
    
    Parameters:
    - x: Input data (must be of even length).
    - dt: Time step.
    
    Returns:
    - Frequency vector and normalized frequency domain representation.
    """
    if len(x) % 2 != 0:
        raise ValueError("Input signal length must be even.")
    
    N = len(x)
    yf = fft(x)
    xf = fftfreq(N, dt)[:N//2]
    norm_amplitude_spectrum = (2.0 / N) * np.abs(yf[:N//2])  # Normalize the amplitude spectrum
    return xf, norm_amplitude_spectrum, yf


def compute_ifft(yf):
    """
    Computes the inverse FFT to reconstruct the time domain signal.
    
    Parameters:
    - yf: The unormalized complex frequency domain representation.
    
    Returns:
    - The reconstructed time domain signal's real and imaginary parts.
    """
    reconstructed_signal = ifft(yf)
    return reconstructed_signal.real, reconstructed_signal.imag



def compute_time_vector(length, dt):
    """
    Computes the time vector.
    
    Parameters:
    - length: Length of the time series.
    - dt: Time step.
    
    Returns:
    - Time vector.
    """
    return np.arange(0, length * dt, dt)


def compute_half_spectrum_for_dataset(data, dt):
    """
    Computes the half spectrum for each sensor in the data.
    
    Parameters:
    - data: Input tensor or numpy array with shape (num_sensors, num_timesteps).
    - dt: Time step.
    
    Returns:
    - Tensor containing the half spectrum for each sensor.
    """
    # Ensure the input is a tensor
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    elif not isinstance(data, torch.Tensor):
        raise TypeError(f"Input must be a torch tensor or numpy array, but got {type(data)}.")

    num_sensors, num_timesteps = data.shape
    half_spectrum = []

    for j in range(num_sensors):
        sensor_data = data[j, :].cpu().detach().numpy()
        _, norm_amplitude_spectrum, _ = compute_normalized_frequency_domain(sensor_data, dt)
        half_spectrum.append(norm_amplitude_spectrum)
    
    # half_spectrum_tensor = torch.tensor(half_spectrum, dtype=torch.float32)
    
    # Convert list of NumPy arrays to a single NumPy array first, then to a tensor
    half_spectrum_array = np.array(half_spectrum, dtype=np.float32)
    half_spectrum_tensor = torch.tensor(half_spectrum_array)
    
    return half_spectrum_tensor


def reconstruct_full_spectrum(half_spectrum):
    """
    Reconstructs the full normalized spectrum from the half spectrum.

    Parameters:
    - half_spectrum: Input half spectrum (torch tensor) with shape (num_sensors, num_freq_bins).

    Returns:
    - Full normalized spectrum (torch tensor) with shape (num_sensors, 2*num_freq_bins).
    """
    if not isinstance(half_spectrum, torch.Tensor):
        raise TypeError(f"Input must be a torch tensor, but got {type(half_spectrum)}.")

    if half_spectrum.is_complex():
        raise ValueError("Input half spectrum must be real-valued.")
    
    num_sensors, num_freq_bins = half_spectrum.shape
    
    # Full spectrum has double the size of the half spectrum
    full_spectrum_size = 2 * num_freq_bins

    # Check the last element is not mirrored
    nyquist_freq = half_spectrum[:, -1:]

    # Mirror the half spectrum, excluding the last element
    mirrored_spectrum = torch.flip(half_spectrum[:, :-1], dims=[1])

    # Concatenate the half spectrum, Nyquist frequency, and the mirrored spectrum
    full_spectrum = torch.cat((half_spectrum, nyquist_freq, mirrored_spectrum), dim=1)

    # Ensure the full spectrum size is correct
    if full_spectrum.shape[1] != full_spectrum_size:
        raise ValueError(
            f"The full spectrum size is incorrect: {full_spectrum.shape[1]}. Expected size is {full_spectrum_size}."
        )

    return full_spectrum



################################
################################
# Plots 
################################
################################

def aug_time_and_plot_time_and_frequency_domain(data, unique_classes, sensors, operation='jitter', sampling_rate=50):
    """
    Plots time and frequency domain representations for specified sensors and classes.

    Parameters:
    - data: Torch tensor containing the time series data with shape (batch_size, number_of_sensors, number_of_timesteps).
    - unique_classes: List of unique classes corresponding to each sample in the batch.
    - sensors: List of sensors to plot.
    - operation: The operation to apply to the data (e.g., 'jitter', 'scaling', 'permutation', 'masking').
    - sampling_rate: Sampling rate of the data in Hz.

    Note:
    - The batch size of the input data must be equal to the number of unique classes for the plots.
    
    Returns:
    - None. The function generates and displays plots.
    """

    # Ensure data is a torch tensor
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input should be a torch tensor.")

    dt = 1.0 / sampling_rate  # time step

    fig, axs = plt.subplots(len(unique_classes), 2, figsize=(15, 5 * len(unique_classes)), constrained_layout=True)

    label_mapping = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS',
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }

    fig.text(0.25, 1.01, 'Time Domain', ha='center', va='center', fontsize=30)
    fig.text(0.75, 1.01, 'Frequency Domain', ha='center', va='center', fontsize=30)

    for i, cls in enumerate(unique_classes):
        label = label_mapping.get(cls, f'Class {int(cls)}')
        for j, sensor in enumerate(sensors):
            sensor_data = data[i, j, :].cpu().detach().numpy()

            time_vector = np.arange(len(sensor_data)) * dt

            xf_orig, norm_yf_orig, _ = compute_normalized_frequency_domain(sensor_data, dt)

            if operation == 'jitter':
                modified_data = jitter(data[i, j, :], sigma=0.6).cpu().detach().numpy()
                affected_indices = []
            elif operation == 'scaling':
                modified_data = scaling(data[i, j, :], sigma=1.1).cpu().detach().numpy()
                affected_indices = []
            elif operation == 'permutation':
                modified_data, affected_indices = permutation(data[i, j, :], max_segments=5, seg_mode="random")
                modified_data = modified_data.cpu().detach().numpy()
            elif operation == 'masking':
                modified_data, affected_indices = masking(data[i, j, :], keepratio=0.9, mask='binomial')
                modified_data = modified_data.cpu().detach().numpy()
            else:
                raise ValueError(f"Operation '{operation}' not supported.")

            xf_mod, norm_yf_mod, _ = compute_normalized_frequency_domain(modified_data, dt)

            axs[i, 0].plot(time_vector, sensor_data, label='Original')
            axs[i, 0].plot(time_vector, modified_data, label=f'{operation.capitalize()}', linestyle='dashed')
            axs[i, 0].set_title(f'{label} - {sensor}')
            axs[i, 0].set_xlabel('Time [s]')
            axs[i, 0].set_ylabel('Amplitude')
            axs[i, 0].legend()

            axs[i, 1].plot(xf_orig, norm_yf_orig, label='Original', alpha=0.5)
            axs[i, 1].plot(xf_mod, norm_yf_mod, label=f'{operation.capitalize()}', linestyle='dashed', alpha=0.5)
            axs[i, 1].set_title(f'{label} - {sensor}')
            axs[i, 1].set_xlabel('Frequency [Hz]')
            axs[i, 1].set_ylabel('Magnitude')
            axs[i, 1].legend()

    plt.subplots_adjust(top=0.85)
    plt.show()
    
    
def aug_freq_and_plot_freq_time_domain(data, unique_classes, sensors, operation='remove_frequency', 
                                       remove_ratio=0.0, perturb_ratio=0.0, sampling_rate=50):
    """
    Applies frequency domain augmentations to the input data and plots the frequency domain representations.
    
    Parameters:
    - data: Torch tensor containing the time series data with shape (batch_size, number_of_sensors, number_of_timesteps).
    - unique_classes: List of unique classes corresponding to each sample in the batch.
    - sensors: List of sensor indices to plot (indices should match the sensor columns in the data tensor).
    - operation: The operation to apply to the data (options: 'remove_frequency', 'add_frequency', 'both').
    - remove_ratio: Ratio of frequencies to remove for the 'remove_frequency' operation.
    - perturb_ratio: Ratio of frequencies to perturb for the 'add_frequency' operation.
    - sampling_rate: Sampling rate of the data in Hz.
    
    Note:
    - The batch size of the input data must be equal to the number of unique classes for the plots.
    - All frequency augmentations are done to the amplitude of the half spectrum, which is purely real.
    
    Returns:
    - None. The function generates and displays plots.
    """
    
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input should be a torch tensor.")
    
    dt = 1.0 / sampling_rate  # time step

    fig, axs = plt.subplots(len(unique_classes), 1, figsize=(15, 5 * len(unique_classes)), constrained_layout=True)

    label_mapping = {
        1: 'WALKING',
        2: 'WALKING_UPSTAIRS',
        3: 'WALKING_DOWNSTAIRS',
        4: 'SITTING',
        5: 'STANDING',
        6: 'LAYING'
    }

    fig.text(0.5, 1.01, 'Frequency Domain', ha='center', va='center', fontsize=30)

    for i, cls in enumerate(unique_classes):
        label = label_mapping.get(cls, f'Class {int(cls)}')
        for j, sensor in enumerate(sensors):
            sensor_data = data[i, j, :].cpu().detach().numpy()

            xf, norm_yf_amplitude, yf_complex = compute_normalized_frequency_domain(sensor_data, dt)
            yf_complex_half = yf_complex[:len(norm_yf_amplitude)]
            
            # Convert the normalized half spectrum to a torch tensor
            half_yf_amplitude_tensor = torch.tensor(norm_yf_amplitude, dtype=torch.float32)

            if operation == 'remove_frequency':
                modified_half_yf_amplitude_tensor = remove_frequency(half_yf_amplitude_tensor, remove_ratio)
            elif operation == 'add_frequency':
                modified_half_yf_amplitude_tensor = add_frequency(half_yf_amplitude_tensor, perturb_ratio)
            elif operation == 'both':
                modified_half_yf_amplitude_tensor = add_frequency(remove_frequency(half_yf_amplitude_tensor, remove_ratio), perturb_ratio)
            else:
                raise ValueError(f"Operation '{operation}' not supported.")

            axs[i].plot(xf, norm_yf_amplitude * len(sensor_data), label='Original')  # Rescale back for display purposes
            axs[i].plot(xf, modified_half_yf_amplitude_tensor.cpu().detach().numpy() * len(sensor_data), label=f'{operation.capitalize()}')
            axs[i].set_title(f'{label} - {sensor}')
            axs[i].set_xlabel('Frequency [Hz]')
            axs[i].set_ylabel('Magnitude')
            axs[i].legend()

    plt.subplots_adjust(top=0.85)
    plt.show()
    
    
    
def plot_model_metrics(model):
    """
    Plots the training and validation metrics for the given model.
    
    Parameters:
    - model: The model containing the training and validation logs.
    
    This function creates a 3x3 grid of subplots showing the following metrics:
    1. Backbone Time Encoder Loss (Training and Validation)
    2. Backbone Freq Encoder Loss (Training and Validation)
    3. Backbone Cross-Modal Loss (Training and Validation)
    4. Total Loss (Training and Validation)
    5. Backbone Loss (Training and Validation)
    6. Projection Head Loss (Training and Validation)
    7. Accuracy (Training and Validation)
    8. Recall (Training and Validation)
    9. F1 Score (Training and Validation)
    """
    fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=False)
    ax = axs.ravel()

    epochs = np.arange(1, len(model.val_loss_log) + 1, 1)

    # Backbone loss components. Their sum results in the backbone total loss

    ax[0].plot(epochs, model.train_loss_time_enc, label="Training")
    ax[0].plot(epochs, model.val_loss_time_enc, label="Validation")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].set_title('Backbone Time Encoder')
    ax[0].legend()

    ax[1].plot(epochs, model.train_loss_freq_enc, label="Training")
    ax[1].plot(epochs, model.val_loss_freq_enc, label="Validation")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].set_title('Backbone Freq Encoder')
    ax[1].legend()

    ax[2].plot(epochs, model.train_loss_consist, label="Training")
    ax[2].plot(epochs, model.val_loss_consist, label="Validation")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Loss")
    ax[2].set_title('Backbone Cross-Modal')
    ax[2].legend()

    ### Losses for Backbone + Projection Head

    # The total loss is the backbone loss + the projection head loss
    ax[3].plot(epochs, model.train_loss_log, label="Training") 
    ax[3].plot(epochs, model.val_loss_log, label="Validation")
    ax[3].set_xlabel("Epochs")
    ax[3].set_ylabel("Loss")
    ax[3].set_title('Total Loss')
    ax[3].legend()

    # The backbone loss is a decimal constant that weighs the importance of the encoder loss and the consistency loss. 
    # Thats why its lower than the other two
    ax[4].plot(epochs, model.train_loss_backbone, label="Training")
    ax[4].plot(epochs, model.val_loss_backbone, label="Validation")
    ax[4].set_xlabel("Epochs")
    ax[4].set_ylabel("Loss")
    ax[4].set_title('Backbone Loss')
    ax[4].legend()

    ax[5].plot(epochs, model.train_loss_proj_head, label="Training")
    ax[5].plot(epochs, model.val_loss_proj_head, label="Validation")
    ax[5].set_xlabel("Epochs")
    ax[5].set_ylabel("Loss")
    ax[5].set_title('Projection Head Loss')
    ax[5].legend()

    ### Metrics for Backbone + Projection Head

    ax[6].plot(epochs, model.train_accuracy_log, label="Training")
    ax[6].plot(epochs, model.val_accuracy_log, label="Validation")
    ax[6].set_xlabel("Epochs")
    ax[6].set_ylabel("Accuracy")
    ax[6].set_title('Backbone + Projection Head')
    ax[6].legend()

    ax[7].plot(epochs, model.train_recall_log, label="Training")
    ax[7].plot(epochs, model.val_recall_log, label="Validation")
    ax[7].set_xlabel("Epochs")
    ax[7].set_ylabel("Recall")
    ax[7].set_title('Backbone + Projection Head')
    ax[7].legend()

    ax[8].plot(epochs, model.train_f1_log, label="Training")
    ax[8].plot(epochs, model.val_f1_log, label="Validation")
    ax[8].set_xlabel("Epochs")
    ax[8].set_ylabel("F1 metric")
    ax[8].set_title('Backbone + Projection Head')
    ax[8].legend()

    plt.tight_layout()
    plt.show()