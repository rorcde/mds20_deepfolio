import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils_data

from utils import compute_integral_mc, compute_integral_li

PADDING_CONST = 0

class IntensityNetwork(nn.Module):
    """
    Neural network used to generate conditional intensity functions.

    Denominations:
        B - batch size,
        S - sequence length,
        E - embedding dimension (d_model for the Transformer),
        N - number of samples (for Monte-Carlo integration),
        K - number of event types
    Paper references:
        Zuo et al. "Transformer Hawkes Process" - https://arxiv.org/abs/2002.09291
    """

    def __init__(self, hidden_size, n_event_types, device):
        """
        Input:
            hidden_size (int) - size of the hidden dimension (d_model),
            n_event_types (int) - number of possible event types in the data,
            device - current device
        """
        super(IntensityNetwork, self).__init__()

        self.n_events = n_event_types
        self.device = device

        # accounts for "history" and "base" (through bias) terms in eq.(6) of the paper
        self.linear = nn.Linear(hidden_size, n_event_types)

        self.softplus = nn.Softplus(threshold=10)

    def generate_type_mask(self, events):
        """
        Function to generate one-hot encoding mask for the event types, needed to compute
        type-specific intensity function.

        Input:
            events (B, S) - sequence of event types
        Output:
            type_mask (B, S, K) - one-hot mask for the event types
        """
        bs, ls = events.size()

        type_mask = torch.zeros(bs, ls, self.n_events, device=self.device)
        for k in range(self.n_events):
            type_mask[:, :, k] = (events == k + 1).bool().to(self.device)
        return type_mask
    
    def forward(self, hidden_states, events, current_inf=None, mc_trick=False):
        """
        Input:
            hidden_states (B, S, E) - hidden states of the Transformer,
            events (B, S) - sequence of event types,
            current_inf (B, S-1, N) - "current" influence in the eq. (6) of the paper,
            mc_trick (bool) - indicator showing whether vectorization trick for the Monte-Carlo integration is active
        Output:
            conditional_lambda (B, S) - conditional intensity function
        """
        intensity_terms = self.linear(hidden_states)
        type_mask = self.generate_type_mask(events)

        if mc_trick:
            # this is a trick for Monte-Carlo integration, which allows to vectorize
            # computation of (num_samples) intensity functions instead of making a loop

            assert current_inf is not None, "current influence cannot be None when mc_trick is True"

            intensity_terms = (intensity_terms[:, :-1, :] * type_mask[:, :-1, :]).sum(dim=2, keepdim=True)
            continious_intensity = self.softplus( intensity_terms + current_inf )
            conditional_lambda = continious_intensity.mean(dim=2)
        else:
            if current_inf is not None:
                intensity_terms += current_inf
            continious_intensity = self.softplus(intensity_terms)

            # (continious_intensity * type_mask) gets type-specific instensity function (eq. (6))
            # after summation along the 2nd dimension, conditional intensity function is obtained
            conditional_lambda = (continious_intensity * type_mask).sum(dim=2)
        
        return conditional_lambda

class HawkesTransformer(nn.Module):
    """
    Main model - Transformer Hawkes Process (THP). Thorough explanation of the model and its architecture can be found either on the
    GitHub or in the report / original paper.

    Denominations:
        B - batch size,
        S - sequence length,
        E - embedding dimension,
        K - number of event types
    Paper references:
        Zuo et al. "Transformer Hawkes Process" - https://arxiv.org/abs/2002.09291
    """

    def __init__(self, n_event_types, device, d_model=512, n_heads=8, dim_feedforward=2048, n_layers=6, dropout=0.1, activation='relu'):
        """
        Input:
            n_event_types (int) - number of event types in the data,
            device - current device,
            d_model (int) - size of the embedding dimension,
            n_heads (int) - number of heads in the Multihead Attention module,
            dim_feedforward (int) - size of the feedforward network hidden dimension,
            n_layers (int) - number of Transformer Encoder layers,
            dropout (float) - dropout rate,
            activation (string) - activation function for the feedforward network (relu or gelu)
        """
        super(HawkesTransformer, self).__init__()

        self.n_events = n_event_types
        self.d_model = d_model
        self.device = device

        # initialize div term for temporal encoding
        self.init_temporal_encoding()

        # event type embedding
        self.event_embedding = nn.Embedding(n_event_types + 1, d_model, padding_idx=PADDING_CONST)

        # transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, activation)
        self.transformer_layers = nn.TransformerEncoder(encoder_layer, n_layers)

        # linear transformation of hidden states ("history" and "base" terms in eq.(6) of the THP paper) to
        # type specific intensity
        self.intensity_layer = IntensityNetwork(d_model, n_event_types, self.device)

        # output prediction layers
        self.time_predictor  = nn.Linear(d_model, 1, bias=False)
        self.event_predictor = nn.Linear(d_model, n_event_types, bias=False)

        # small constant
        self.eps = torch.tensor([1e-9], device=self.device)

    def generate_subsequent_mask(self, seq):
        """
        Function to generate masking for the subsequent information in the sequences (masked self-attention).

        Input:
            seq (B, S) - batch of sequences
        Output:
            mask (S, S) - subsequent mask
        """
        bs, ls = seq.size()
        subsequent_mask = torch.triu( torch.ones(ls, ls, device=self.device, dtype=torch.bool), diagonal=1 )
        
        return subsequent_mask
    
    def generate_key_padding_mask(self, seq):
        """
        Masking the padded part of the sequence.

        Input:
            seq (B, S) - batch of sequences
        Output:
            mask (B, S) - padding mask
        """
        padding_mask = seq.eq(PADDING_CONST)

        return padding_mask

    def init_temporal_encoding(self):
        """
        Initializing the internal temporal encoding tensors.
        """
        encoding_constant = torch.tensor(10000.0)

        # for better numerical stability
        self.te_div_term = torch.exp(2.0 * (torch.arange(0, self.d_model) // 2) * -torch.log(encoding_constant) / self.d_model).to(self.device)
  
    def temporal_encoding(self, t, non_padded_mask):
        """
        Function to perform the temporal encoding on input timestamps.

        Input:
            t (B, S) - batch of time sequences,
            non_padded_mask (B, S) - binary mask indicating whether element is a padding (True) or not (False)
        Output:
            t_enc (B, S) - temporal encoding of the time sequences
        """
        temporal_enc = t.unsqueeze(-1) * self.te_div_term

        temporal_enc[:, :, 0::2] = torch.sin(temporal_enc[:, :, 0::2])
        temporal_enc[:, :, 1::2] = torch.cos(temporal_enc[:, :, 1::2])

        return temporal_enc * non_padded_mask.unsqueeze(-1)
    
    def log_likelihood(self, hidden_states, cond_lam, time, events, alpha, integral='mc'):
        """
        Function to compute log likelihood for the sequence. Beware that our aim is to maximize this function,
        hence to use it as a loss, we need to take "minus" the output of this function.

        Input:
            hidden_states (B, S, E) - hidden states of the network,
            cond_lam (B, S, F) - conditional intensity function,
            time (B, S) - ground truth for times,
            events (B, S) - ground truth for event types,
            alpha (int) - scaling constant,
            integral (string) - method of integration: either Monte-Carlo (unbiased) or linear interpolation (biased)
        Output:
            log_likelihood (float) - log likelihood for the batch of sequences
        """
        
        src_padding_mask = self.generate_key_padding_mask(events)

        # # compute event log-likelihood
        event_part = cond_lam + self.eps
        event_part.masked_fill_(src_padding_mask, 1.0)
        event_part = event_part.log()
        event_part = event_part.sum(dim=1)

        # # compute non-event log-likelihood
        if integral == 'mc':
            non_event_part = compute_integral_mc(self.intensity_layer, hidden_states, src_padding_mask, time, events, alpha)
        else:
            non_event_part = compute_integral_li(cond_lam, time, src_padding_mask)
        non_event_part = non_event_part.sum(dim=1)

        # # compute total log-likelihood
        log_likelihood = (event_part - non_event_part).sum()

        return log_likelihood

    def time_error(self, time_pred, time):
        """
        Function to compute mean squared error for time predictions.

        Input:
            time_pred (B, S) - time predictions,
            time (B, S) - ground truth for times
        Output:
            time_error (float) - time prediction error for the whole batch
        """

        time_ground_truth = time[:, 1:] - time[:, :-1]
        time_pred = time_pred[:, :-1]

        time_error = nn.MSELoss(reduction='sum')(time_pred, time_ground_truth)
        return time_error

    def event_error(self, event_logit, events):
        """
        Function to compute cross entropy loss for the event type classification.

        Input:
            event_logit (B, S, K) - event type logits,
            events (B, S) - ground truth for event types
        Output:
            event_error (float) - cross entropy loss for the batch
        """

        event_ground_truth = events[:, 1:] - 1
        event_logit = event_logit[:, :-1, :]

        event_error = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)(event_logit.transpose(1, 2), event_ground_truth)
        return event_error
    
    def forward(self, time, events):
        """
        Input:
            time (B, S) - input sequence of event timestamps,
            events (B, S) - input sequence of event types
        Output:
            h (B, S, E) - hidden states of the Transformer,
            cond_lam (B, S) - conditional intensity function,
            time_pred (B, S) - time predictions,
            event_logit (B, S, K) - event type logits (need to use softmax followed by argmax to obtain predictions)
        """

        # generate masks
        src_key_padding_mask = self.generate_key_padding_mask(events)
        src_non_padded_mask = ~src_key_padding_mask
        src_mask = self.generate_subsequent_mask(events)

        # perform encodings
        temp_enc  = self.temporal_encoding(time, src_non_padded_mask)
        event_enc = self.event_embedding(events)

        # make pass through transformer encoder layers
        x = event_enc + temp_enc
        h = self.transformer_layers(x.permute(1, 0, 2), mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        h = h.permute(1, 0, 2)

        # obtain conditional intensity function
        cond_lam = self.intensity_layer(h, events)

        # make predictions
        time_pred  = self.time_predictor(h).squeeze(2) * src_non_padded_mask
        event_logit = self.event_predictor(h) * src_non_padded_mask.unsqueeze(-1)

        return h, cond_lam, time_pred, event_logit