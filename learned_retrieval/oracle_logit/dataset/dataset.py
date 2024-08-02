from torch.utils.data import Dataset
import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

class CompletionContextDataset(Dataset):
    def __init__(self, input_data):
        self.data = input_data[['completion_content', 'context_files', 'avg_cross_entropy', 'EMs']]
        self.data['context_files'] = self.data['context_files'].apply(lambda x: x[0]['content'])

    def mean_std_norm(self, do_test=False, mean_=None, std_=None, do_sigmoid=False, do_clip=False):
        if not do_test:
            mean_ = self.data['avg_cross_entropy'].mean()
            std_ = self.data['avg_cross_entropy'].std()
        
        self.data['norm_cross_entropy'] = (self.data['avg_cross_entropy'] - mean_) / std_

        if do_sigmoid:
            self.data['norm_cross_entropy'] = sigmoid(self.data['norm_cross_entropy'])
        
        if do_clip:
            self.data['norm_cross_entropy'] = np.clip(self.data['norm_cross_entropy'], 0, 1)

        return mean_, std_

    def min_max_norm(self, do_test=False, min_=None, max_=None, do_clip=False):
        if not do_test:
            min_ = self.data['avg_cross_entropy'].min()
            max_ = self.data['avg_cross_entropy'].max()

        self.data['norm_cross_entropy'] = (self.data['avg_cross_entropy'] - min_) / (max_ - min_)

        if do_clip:
            self.data['norm_cross_entropy'] = np.clip(self.data['norm_cross_entropy'], 0, 1)
        
        return min_, max_
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        completion = self.data.iloc[idx]['completion_content']
        context = self.data.iloc[idx]['context_files']
        norm_cross_entropy = self.data.iloc[idx]['norm_cross_entropy']

        return completion, context, norm_cross_entropy