from torch.utils.data import Dataset


class DatasetPEMS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, index):
        week_sample = self.data['week'][index]
        day_sample = self.data['day'][index]
        recent_sample = self.data['recent'][index]
        label = self.data['target'][index]

        return week_sample, day_sample, recent_sample, label
