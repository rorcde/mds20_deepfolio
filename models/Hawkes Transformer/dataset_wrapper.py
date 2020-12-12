import pickle

import torch
import torch.utils.data as utils_data

import numpy as np

from utils import make_consequent_slices, preprocess_sliced_data

class LobDataset(utils_data.Dataset):
    """
    Wrapper for LOB dataset.
    """

    def __init__(self, raw_data, slice_inputs=True, preprocess=True, slicing_window=3000, omit_last=True, ts_scaling=1e-3):
        """
        Input:
            raw_data (N, 3)    - either raw dataset with N samples and 3 features (interarrival_time, time_from_start, event_type)
                     (B, S, 3) - or batch of sequences
            slice_inputs (bool) - indicator showing whether the data needs to be sliced into sequences or not,
            preprocess (bool) - indicator showing whether sliced data needs to be preprocessed,
            slicing_window (int) - size of the slices,
            omit_last (bool) - omit last part of the data that is smaller than slicing_window,
            ts_scaling (float) - scaling constant, which converts timestamps to seconds from its original unit
        """

        self.slice_inputs = slice_inputs
        self.preprocess = preprocess
        self.slicing_window = slicing_window
        self.omit_last = omit_last
        self.ts_scaling = ts_scaling

        self.sliced_data = make_consequent_slices(raw_data, slicing_window, omit_last) if slice_inputs else raw_data
        self.sliced_data = preprocess_sliced_data(self.sliced_data, ts_scaling) if preprocess else self.sliced_data

    def __len__(self):
        return len(self.sliced_data)
    
    def expand(self, external_data):
        """
        Expand dataset.

        Input:
            external_data (N, 3) / (B, S, 3) - additional data to be added
        """
        
        additional_data = make_consequent_slices(external_data, self.slicing_window, self.omit_last) if self.slice_inputs else external_data
        additional_data = preprocess_sliced_data(additional_data, self.ts_scaling) if self.preprocess else additional_data

        self.sliced_data = np.concatenate((self.sliced_data, additional_data), axis=0)
    
    def __getitem__(self, index):

        event_time = torch.tensor(self.sliced_data[index, :, 1], dtype=torch.float)
        event_type = torch.LongTensor(self.sliced_data[index, :, 2].astype(int))

        return event_time, event_type

class Dataset(utils_data.Dataset):
    """
    Wrapper for NHP financial dataset.
    """

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

        event_type = torch.LongTensor(self.event_type[index].long() + 1)
        event_time = torch.Tensor(self.event_time[index])
        seq_len = torch.tensor(len(event_type))
        event_last_time = torch.sum(event_time)

        X = torch.stack((event_time, event_type), dim=1)
        
        return event_time, event_type
