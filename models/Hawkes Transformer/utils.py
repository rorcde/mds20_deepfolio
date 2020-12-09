import torch

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