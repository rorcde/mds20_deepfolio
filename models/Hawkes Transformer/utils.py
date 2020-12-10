import torch
import numpy as np

def make_consequent_slices(raw_data, slicing_window, omit_last=True, timestamp_scaling=1e-3):
    """
    Function to cut the one long sequence of events into smaller ones.
    
    Input:
        raw_data (N, F) - raw dataset with N samples and F features,
        slicing_window (int) - size of the slices,
        omit_last (bool) - omit last part of the data that is smaller than slicing_window,
        timestamp_scaling (float) - scaling constant, which converts timestamps to seconds from its original unit
    Output:
        sequential_slices (B, S, F) - sliced dataset with B batches of length S with F features
    """
    
    N = raw_data.shape[0]

    slices = []
    for i in range(0, N, slicing_window):
        slices.append( raw_data[i:i + slicing_window].copy() )
        slices[-1][:, :2] *= timestamp_scaling
        slices[-1][0, :2] = 0.
        slices[-1][:, 2] += 1
    
    if omit_last:
        slices.pop()
    elif slices[-1].shape[0] != slicing_window:
        zero_padding = np.zeros(slices[-1].shape[0] - slicing_window, slices[-1].shape[1])
        slices[-1] = np.concatenate((slices[-1], zero_padding), axis=0)
    
    sequential_slices = np.stack(slices, axis=0)
    return sequential_slices

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
