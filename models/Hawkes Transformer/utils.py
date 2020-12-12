import torch

import numpy as np
import random

def fix_seed(seed):
    """
    Fix seed for reproducibility.

    Input:
      seed (int) - desired seed
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def preprocess_sliced_data(seq_data, ts_scaling=1e-3):
    """
    Function for sequence preprocessing - scaling timestamps, removing the time shift.

    Input:
        seq_data (B, S, 3) - batch of sequences of size B, sequence lengths S and 3 features (interarrival_time, time_from_start, event_type),
        ts_scaling (float) - scaling constant, which converts timestamps to seconds from its original unit
    Output:
        preprocessed_seqs (B, S, 3) - preprocessed sequences
    """
    
    preprocessed_seqs = seq_data

    preprocessed_seqs[:, :, :2] *= ts_scaling
    preprocessed_seqs[:, 0, 0]   = 0.
    preprocessed_seqs[:, :, 2]  += 1

    for n_batch in range(len(preprocessed_seqs)):
        preprocessed_seqs[n_batch, :, 1] -= preprocessed_seqs[n_batch, 0, 1]

    return preprocessed_seqs

def make_consequent_slices(raw_data, slicing_window, omit_last=True):
    """
    Function to cut the one long sequence of events into smaller ones.
    
    Input:
        raw_data (N, 3) - raw dataset with N samples and 3 features (interarrival_time, time_from_start, event_type),
        slicing_window (int) - size of the slices,
        omit_last (bool) - omit last part of the data that is smaller than slicing_window
    Output:
        sequential_slices (B, S, F) - sliced dataset with B batches of length S with F features
    """
    
    N = raw_data.shape[0]

    slices = []
    for i in range(0, N, slicing_window):
        slices.append( raw_data[i:i + slicing_window].copy() )
    
    if omit_last:
        slices.pop()
    elif slices[-1].shape[0] != slicing_window:
        zero_padding = np.zeros(slices[-1].shape[0] - slicing_window, slices[-1].shape[1])
        slices[-1] = np.concatenate((slices[-1], zero_padding), axis=0)
    
    sequential_slices = np.stack(slices, axis=0)

    return sequential_slices

def check_stopping_criterion(current_criterion_value, best_criterion_value, criterion):
    """
    Function to check criterion changes for early stopping.

    Input:
        current_criterion_value (float) - current criterion value,
        best_criterion_value, (float) - current best criterion value,
        criterion (string) - stopping criterion (either 'min_loss' or 'max_accuracy')
    Output:
        improved (bool) - flag indicating whether criterion value improved compared to the best value
    """

    return current_criterion_value < best_criterion_value if criterion == 'min_loss' else current_criterion_value > best_criterion_value

def compute_integral_mc(intensity_network, hidden_states, src_padding_mask, time, events, alpha=-0.1, n_samples=100):
    """
    Compute integral using Monte Carlo integration.
    """
    # time differences (t_{j} - t{j-1})
    dt = (time[:, 1:] - time[:, :-1]) * (~src_padding_mask[:, 1:])

    # sample t from uniform distribution: 
    # since t \in (t_{j-1}, t_j) which would lead to (t - t_{j-1}) / t_{j-1} \in [0, (t_j - t_{j-1}) / t_{j-1}),
    # we can reformulate this as (t_j - t_{j-1}) / t_{j-1} * u, where u \in [0, 1)
    current_influence = alpha * (dt.unsqueeze(2) / (time[:, :-1] + 1).unsqueeze(2)) * torch.rand([*dt.size(), n_samples], device=hidden_states.device)

    # compute sum( lambda(u_i) ) / N
    mc_intensity = intensity_network(hidden_states, events, current_inf=current_influence, mc_trick=True)

    return mc_intensity * dt

def compute_integral_li(lam, time, src_padding_mask):
    dt = (time[:, 1:] - time[:, :-1]) * (~src_padding_mask[:, 1:])
    dlam = (lam[:, 1:] + lam[:, :-1]) * (~src_padding_mask[:, 1:])

    integral = dt * dlam
    return 0.5 * integral
