# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np

class LogLikelihoodLoss(nn.Module):
    def __init__(self, device=None):
        super(LogLikelihoodLoss, self).__init__()
        self.device = device

    def create_unif_d(self, batch_length, total_time_seqs):

        sim_time_seqs = []
        for tot_time in total_time_seqs:
            sim_time_seqs.append(torch.rand(batch_length).uniform_(0,total_time_seqs[0]))
        sim_time_seqs = torch.stack(sim_time_seqs)
        sim_time_seqs = sim_time_seqs.transpose(1,0)
        if self.device:
            sim_time_seqs = sim_time_seqs.to(device)

        return sim_time_seqs

    def forward(self, model, event_seqs, time_seqs, seqs_length, total_time_seqs, output, batch_first=True):

        batch_size, batch_length = event_seqs.shape
        hidden_t, cell_t, cell_target_t, output_t, decay_t = [torch.squeeze(torch.stack(val), 0) for val in output]
        

        """ Compute log-likelihood of of the events that happened (first term) via sum of log-intensities """

        intensity = model.intensity_layer(hidden_t)
        log_intensity = intensity.log()
        #shape - S * bs * H

        original_loglikelihood = 0
        for idx, (event_seq, seq_len) in enumerate(zip(event_seqs, seqs_length)):
            arr = torch.arange(seq_len)
            pos_events = event_seq[1:seq_len+1]
            original_loglikelihood += log_intensity[arr, idx, pos_events].sum()


        """ Compute log-probabilities of non-events using Monte Carlo method (see Appendix B2) """

        ### 1) Create t ∼ Unif(0, T)
        sim_time_seqs = self.create_unif_d(batch_length, total_time_seqs)

        ### 2) Find intensities for simulated events
        hidden_t_sim = []
        for idx, sim_duration in enumerate(sim_time_seqs):
            _, h_t_sim = model.decay_cell(cell_t[idx], cell_target_t[idx], output_t[idx], decay_t[idx], sim_duration)
            hidden_t_sim.append(h_t_sim)            
        sim_intensity = model.intensity_layer(torch.stack(hidden_t_sim))

        ### 2) Caclulate integral using Monte Carlo method
        simulated_likelihood = 0
        for idx, (total_time, seq_len) in enumerate(zip(total_time_seqs, seqs_length)):
            mc_coefficient = total_time / (seq_len)
            arr = torch.arange(seq_len)
            simulated_likelihood += mc_coefficient * sim_intensity[arr, idx, :].sum(dim=(1,0))

        loglikelihood = original_loglikelihood - simulated_likelihood
        return -loglikelihood


from sklearn.metrics import accuracy_score, mean_squared_error

def evaluate_prediction(model, dataloader, device):
    pred_data = []
    for sample in dataloader:
        event_seqs, time_seqs, total_time_seqs, seqs_length = pad_bos(sample, model.type_size)
        for i in range(len(event_seqs)):
            pred_data.append(predict_event(model, time_seqs[i], event_seqs[i], seqs_length[i], device))
    
    pred_data = np.array(pred_data)
    time_gt, time_pred = pred_data[:,0], pred_data[:,1]
    type_gt, type_pred = pred_data[:,2], pred_data[:,3]

    time_mse_error = mean_squared_error(time_gt, time_pred)
    type_accuracy = accuracy_score(type_gt, type_pred)
    
    return time_mse_error, type_accuracy



def predict_event(model, seq_time, seq_events, seq_lengths, device, hmax = 40,
                     n_samples=1000):
  

        """ Feed the model with sequence and compute decay cell state """

        timestamps = seq_time.cumsum(dim=0).to(device)
        with torch.no_grad():
            model.init_states(1)
            for i in range(seq_lengths):
                c_t, c_target, output, decay = model.LSTM_cell(model.Embedding(seq_events[i].to(device)).unsqueeze(0), model.hidden_decay, 
                                                               model.cell_decay, model.cell_target)

                if i < seq_lengths - 1:
                    c_t = c_t * torch.exp(-decay * seq_time[i, None].to(device)) 
                    h_t = output * torch.tanh(c_t)

            # gt last and one before last event types and times
            last_t, gt_t = timestamps[i], timestamps[i + 1]
            last_type, gt_type = seq_events[i], seq_events[i + 1]
            gt_dt = seq_time[i]


            """ Make prediction for the next event time and type """
            model.eval()
            timestep = hmax / n_samples

            # 1) Compute intensity
            time_between_events = torch.linspace(0, hmax, n_samples + 1).to(device)
            hidden_vals = h_t * torch.exp(-decay * time_between_events[:, None])
            intensity = model.intensity_layer(hidden_vals.to(device))
            

            # 2) Compute density via integral 
            density = torch.cumsum(timestep * intensity.sum(dim=1), dim=0)
            density = intensity.sum(dim=1) * torch.exp(-density)

            # 3) Predict time of the next event via trapeze method
            t = time_between_events * density   
            pred_dt = (timestep * 0.5 * (t[1:] + t[:-1])).sum() 
            
            # 4) Predict type of the event via trapeze method
            P = intensity / intensity.sum(dim=1)[:, None] * density[:, None]  
            pred_type = torch.argmax(timestep * 0.5 * (P[1:] + P[:-1])).sum(dim=0)

            return pred_dt.cpu().numpy(), gt_dt.cpu().numpy(), pred_type.cpu().numpy(), gt_type.cpu().numpy()
