import imp
from torch.utils.data import Dataset
import torch


class CSVDataset(Dataset):
    def __init__(self, data) -> None:
        super(CSVDataset, self).__init__()
        self.result = data
        self.len = len(data['index'])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return {
            'Xi': torch.LongTensor(self.result['index'][index]),
            'Xv': torch.FloatTensor(self.result['value'][index]),
            'label': torch.LongTensor([int(self.result['label'][index])])
        }
