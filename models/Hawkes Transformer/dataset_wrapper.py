import pickle

import torch
import torch.utils.data as utils_data

class Dataset(utils_data.Dataset):

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