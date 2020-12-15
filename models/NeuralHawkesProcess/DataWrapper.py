# -*- coding: utf-8 -*-

import pickle
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class NHPDataset(Dataset):
    ''' 
    Create Dataset for Neural Hawkey Process
    using financial data from original paper
    '''

    def __init__(self, file_path):
        self.event_type = []
        self.event_time = []

        with open(file_path, 'rb') as f:

            if 'dev' in file_path:
                seqs = pickle.load(f, encoding='latin1')['dev']
            elif 'train' in file_path:
                seqs = pickle.load(f, encoding='latin1')['train']
            elif 'test' in file_path:
                seqs = pickle.load(f, encoding='latin1')['test']

            for idx, seq in enumerate(seqs):
                self.event_type.append(torch.Tensor([int(event['type_event']) for event in seq]))
                self.event_time.append(torch.Tensor([float(event['time_since_start']) for event in seq]))

    def __len__(self):
        return len(self.event_type)
    
    def __getitem__(self, index):

        event_type = torch.LongTensor(self.event_type[index].long())[1:]
        event_time = torch.Tensor(self.event_time[index])
        #delta_time = torch.zeros_like(event_time)
        delta_time = event_time[1:] - event_time[:-1]
        
        return delta_time, event_type

    
def prepare_datasets(data_directory, train_size=0.6, val_size=0.2):
    """
    Prepare train, val and test datasets from several files
    with given proportions.
    Input:
        data_directory (str) - path to directory with data in .npy format
        train_size (float) - proportion of the dataset to include in the train split
        val_size (float) - proportion of the dataset to include in the val split
    Output:
        train_dsets (dict) - dictionary with train datasets and correspondig file names
        val_dsets (dict) - dictionary with val datasets and correspondig file names
        test_dsets (dict) - dictionary with test datasets and correspondig file names
    """

    datasets = {}
    for file in os.listdir(data_directory):
      if file.endswith(".npy"):
        datasets[file[:3]] = np.load(data_directory + file)
    
    train_dsets, val_dsets, test_dsets = {}, {}, {}
    for (name, dset) in datasets.items():

        train_part, val_part = int(train_size * len(dset)), int(val_size * len(dset))
        train_dsets[name] = LOBDataset(dset[:train_part])
        val_dsets[name]   = LOBDataset(dset[train_part:train_part + val_part])
        test_dsets[name]  = LOBDataset(dset[train_part + val_part:])

    return train_dsets, val_dsets, test_dsets

class LOBDataset(Dataset):
    """
    Wrapper for LOB dataset.
    """

    def __init__(self, data):
        """
        Input:
            data (B, S, 3) - batch of sequences
        """

        self.event_type = []
        self.event_time = []

        for seq in data:
            self.event_type.append(torch.Tensor(seq[:,2]))
            self.event_time.append(torch.Tensor(seq[:,0])/1000)

    def __len__(self):
        return len(self.event_type)
    
    def __getitem__(self, index):

        event_type = torch.LongTensor(self.event_type[index].long())
        event_time = torch.Tensor(self.event_time[index])
        
        return event_time, event_type

def collate_fn(batch, n_events=2):

      """
      While initializing LSTM we have it read a special beginning-of-stream (BOS) event (k0, t0), 
      where k0 is a special event type and t0 is set to be 0 
      (expanding the LSTM’s input dimensionality by one) see Appendix A.2
      Input:
          batch tuple((batch_size, seq_len)x2) - batch with event types sequence and corresponding inter-arrival times
      Output:
          pad_event_seq (batch_size, seq_len) - padded event types sequence
          pad_time_seq (batch_size, seq_len) - padded event times sequence
      """
      seq_time = torch.stack([sample[0] for sample in batch])
      seq_events = torch.stack([sample[1] for sample in batch])

      pad_event = torch.zeros_like(seq_events[:,0]) + n_events
      pad_time = torch.zeros_like(seq_time[:,0])
      pad_event_seq = torch.cat((pad_event.reshape(-1,1), seq_events), dim=1)
      pad_time_seq = torch.cat((pad_time.reshape(-1,1), seq_time), dim=1)

      return pad_event_seq.long(), pad_time_seq
