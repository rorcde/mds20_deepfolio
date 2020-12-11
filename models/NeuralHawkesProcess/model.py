# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F

class CTLSTMCell(nn.Module):
    """ 
        Continuous-time LSTM cell, which makes updates to cell, cell target, output and decay states
    """
    def __init__(self, hidden_size):

        """ 
        Input:
            hidden_size (int) - model hidden size
        """

        super(CTLSTMCell, self).__init__()
        self.input_g = nn.Linear(2*hidden_size, hidden_size)
        self.forget_g = nn.Linear(2*hidden_size, hidden_size)
        self.output_g = nn.Linear(2*hidden_size, hidden_size)
        self.input_target = nn.Linear(2*hidden_size, hidden_size)
        self.forget_target = nn.Linear(2*hidden_size, hidden_size)
        self.z_gate = nn.Linear(2*hidden_size, hidden_size)
        self.decay_g = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, event, hidden, cell, cell_target):

        """
        Compute decayed cell and hidden states
        Input:
            event (batch_size, hidden_size) - emebedded sequence of event types
            hidden (batch_size, hidden_size) - current hidded state (h(t)) 
            cell (batch_size, hidden_size) - current cell state (c(t)) 
            cell_target (batch_size, hidden_size) - current cell target state (c_t(t)) 
        Output:
            cell (batch_size, hidden_size) - updated cell state (c_{i+1})
            cell_target (batch_size, hidden_size) - updated cell target state (c_t_{i+1})
            output_gate (batch_size, hidden_size) - updated output gate state (o_{i+1})
            decay_cell (batch_size, hidden_size) - updated decay state (delta_{i+1})
        """

        # pass current states through CTLSTM layers to obtain updated ones
        input = torch.cat((event, hidden), dim=1)
        input_gate = torch.sigmoid(self.input_g(input))
        forget_gate = torch.sigmoid(self.forget_g(input))
        input_gate_target = torch.sigmoid(self.input_target(input))
        forget_gate_target = torch.sigmoid(self.forget_target(input))
        output_gate = torch.sigmoid(self.output_g(input))  
        z_gate = torch.tanh(self.z_gate(input))
        decay_cell = F.softplus(self.decay_g(input))

        cell = forget_gate * cell + input_gate * z_gate
        cell_target = input_gate_target * cell_target + input_gate_target * z_gate

        return cell, cell_target, output_gate, decay_cell


class NHPModel(nn.Module):
    """ Continuous time LSTM network with decay function """
    def __init__(self, hidden_size,  device, n_events=2):

        """
        Input:
            hidden_size (int) - model hidden size,
            n_events (int) - number of event types in the data,
            device - current device,
        """

        super(NHPModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_events = n_events
        self.device = device

        # continuous-time LSTM cell 
        self.CTLSTM_cell = CTLSTMCell(self.hidden_size)

        # event type embedding
        self.Embedding = nn.Embedding(self.n_events+1, self.hidden_size)

        # layer to compute intensity
        self.intensity_layer = nn.Sequential(
                                  nn.Linear(self.hidden_size, self.n_events),
                                  nn.Softplus())

        # layers to predict time and type of next event
        self.time_predictor  = nn.Linear(hidden_size, 1, bias=False)
        self.event_predictor = nn.Linear(hidden_size, n_events, bias=False)

    def init_states(self, batch_size):

        """      
        Initialize the hidden decayed state and the cell states
        """

        self.hidden_decay = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=self.device)
        self.cell = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=self.device)
        self.cell_decay = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=self.device)
        self.cell_target = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=self.device)

    def decay_cell(self, cell, cell_target, output_gate, decay_cell, T):

        """
        Compute decayed cell and hidden states
        Input:
            cell (batch_size, hidden_size) - cell state
            cell_target (batch_size, hidden_size) - target cell state 
            output_gate - CTLSTM output gate
            decay_cell - rate of decay
            T (float) - interval between events
        Output:
            hidden_decay (batch_size, hidden_size) - decayed (current) hidden state
            cell_decay (batch_size, hidden_size) - decayed (current) cell state
        """

        cell_decay = cell_target + (cell - cell_target) * torch.exp(-decay_cell * T.view(-1,1))
        hidden_decay = output_gate * torch.tanh(cell)

        return cell_decay, hidden_decay

    
    def LogLikelihoodLoss(self, intensity, event_times):

        """
        Compute log-likelihood loss for the sequence using method proposed in Appendix B1-B2
        Input:
            intensity (batch_size, seq_len, n_events) - intensity values for event sequence
            time_seq (batch_size, seq_len) - input sequence of event inter-arriaval times
        Output:
            LLH (float) - log-likelihood for sequence (minus because we maximazing it)
        """

        # Compute log-likelihood of of the events that happened (first term) via sum of log-intensities
        # sum over > 1 to avoid BOS event 
        original_loglikelihood = intensity[:,1:].log().sum(dim=1)

        #Compute log-probabilities of non-events (second term) using Monte Carlo method

        # create simulated inter-arrival times 
        sim_times = create_unifrom_d(event_times, self.device)

        # compute simulated hidden states via current model parameters
        hidden_t_sim = []
        for idx, sim_duration in enumerate(range(sim_times.shape[1])):
            _, h_t_sim = model.decay_cell(self.cell_t[idx], self.cell_target_t[idx], self.output_t[idx], self.decay_t[idx], sim_times[:,idx])
            hidden_t_sim.append(h_t_sim)

        # find simulated intensity
        sim_intensity = model.intensity_layer(torch.stack(hidden_t_sim))

        # calculate integral using Monte-Carlo methon (see Appendix B2)
        tot_time_seqs, seq_len = event_times.sum(dim=1), event_times.shape[1]
        mc_coef = (tot_time_seqs / seq_len).to(self.device)
        simulated_likelihood = sim_intensity.sum(dim=(0,2)) * mc_coef
        
        # sum over batch
        LLH = (original_loglikelihood.sum(dim=1) - simulated_likelihood).sum()
        return - LLH

    def event_loss(self, events_pred, events_gt):

          """
          Compute cross entropy loss for the event type prediction
          Input:
              events_pred (batch_size, seq_len, n_events) - event type probabilities prediction
              events_gt (batch_size, seq_len) - ground truth for event types
          Output:
              event_error (float) - cross entropy loss for the batch
          """

          events_gt = events_gt[:, 1:]
          events_pred = events_pred[:, :-1]
          loss = nn.CrossEntropyLoss()(torch.transpose(events_pred, 1,2), events_gt)

          return loss
        
    def time_loss(self, times_pred, times_gt):
          """
          Compute mean squared error loss for time predictions
          Input:
              times_pred (batch_size, seq_len) - time predictions
              times_gt (batch_size, seq_len) - ground truth for times
          Output:
              time_loss (float) - time prediction loss 
          """

          times_gt = times_gt[:,1:]
          times_pred = times_pred[:,:-1]
          time_loss = nn.MSELoss()(times_pred, times_gt)

          return time_loss
        
    def forward(self, event_seq, time_seq):

        """
        Input:
            time_seq (batch_size, seq_len) - input sequence of event inter-arriaval times
            event_seq (batch_size, seq_len) - input sequence of event types
        Output:
            intensity (batch_size, seq_len, n_events) - intensity values for event sequence
            time_pred (batch_size, seq_len) - next event time predictions
            event_pred (batch_size, seq_len, n_events) - next event type predictions
        """

        batch_size, batch_len = event_seq.shape

        # initialize hidden states
        self.init_states(batch_size)
        self.hidden_t, self.cell_t, self.cell_target_t, self.output_t, self.decay_t = [], [], [], [], []

        for i in range(batch_len):
            
            # update CTLSTM states
            self.cell, self.cell_target, output, decay = self.CTLSTM_cell(self.Embedding(event_seq[:,i]), 
                                                                   self.hidden_decay, self.cell_decay, self.cell_target)
            
            # update decayed cell and hidden states
            self.cell_decay, self.hidden_decay = self.decay_cell(self.cell, self.cell_target, output, decay, time_seq[:,i])
            
            # save some values for futher computations
            self.hidden_t.append(self.hidden_decay)
            self.cell_t.append(self.cell)
            self.cell_target_t.append(self.cell_target)
            self.output_t.append(output)
            self.decay_t.append(decay)
        
        # obtain sequence intensity
        self.hidden_t = torch.transpose(torch.stack(self.hidden_t),0,1)
        intensity = self.intensity_layer(self.hidden_t)

        # make predictions
        time_pred  = self.time_predictor(self.hidden_t).squeeze(2) 
        event_pred = self.event_predictor(self.hidden_t) 

        return  intensity, time_pred, event_pred
