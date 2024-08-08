from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

class BaseCompletionContextDataset(Dataset):
    def __init__(self, input_data: pd.DataFrame):
        self.data = []
        self.data_pairs = []
        self.initialize_data(input_data)
        self.data = pd.DataFrame(self.data, columns=self.get_columns())

    def initialize_data(self, input_data: pd.DataFrame):
        raise NotImplementedError("Subclasses must implement this method")

    def get_columns(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def create_instance(dataset_type: str, input_data: pd.DataFrame):
        if dataset_type == 'logit':
            return LogitCompletionContextDataset(input_data)
        elif dataset_type == 'em':
            return EMCompletionContextDataset(input_data)
        elif dataset_type == 'pos_neg_pairs':
            return PairsCompletionContextDataset(input_data)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

class LogitCompletionContextDataset(BaseCompletionContextDataset):
    def initialize_data(self, input_data: pd.DataFrame):
        for i, item in input_data.iterrows():
            completion = item['completion_content']
            avg_cross_entropy = np.asarray(item['avg_cross_entropy'])
            em = np.asarray(item['EMs'])
            contexts = np.asarray([x[0]['content'] for x in item['context_files']])

            for em_i, avg_ce_i, context_i in zip(em, avg_cross_entropy, contexts):
                self.data.append((completion, context_i, avg_ce_i, em_i))

    def get_columns(self):
        return ['completion', 'context', 'avg_cross_entropy', 'EM']

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

    def __getitem__(self, idx):
        completion = self.data.iloc[idx]['completion']
        context = self.data.iloc[idx]['context']
        norm_cross_entropy = self.data.iloc[idx]['norm_cross_entropy']
        return completion, context, norm_cross_entropy
        
    def __len__(self):
        return len(self.data)
    
class EMCompletionContextDataset(BaseCompletionContextDataset):
    def initialize_data(self, input_data: pd.DataFrame):
        for i, item in input_data.iterrows():
            completion = item['completion_content']
            em = np.asarray(item['EMs'])
            contexts = np.asarray([x[0]['content'] for x in item['context_files']])

            for em_i, context_i in zip(em, contexts):
                self.data.append((completion, context_i, float(em_i)))

    def get_columns(self):
        return ['completion', 'context', 'EM']

    def __getitem__(self, idx):
        completion = self.data.iloc[idx]['completion']
        context = self.data.iloc[idx]['context']
        norm_cross_entropy = self.data.iloc[idx]['EM']
        return completion, context, norm_cross_entropy
    
    def __len__(self):
        return len(self.data)
    
class PairsCompletionContextDataset(BaseCompletionContextDataset):
    def initialize_data(self, input_data: pd.DataFrame):
        for i, item in input_data.iterrows():
            completion = item['completion_content']
            em = np.asarray(item['EMs'])
            contexts = np.asarray([x[0]['content'] for x in item['context_files']])

            for em_i, context_i in zip(em, contexts):
                self.data.append((completion, context_i, float(em_i)))

            positive_indices = np.where(em == 1)[0]
            negative_indices = np.where(em == 0)[0]

            positive_contexts = contexts[positive_indices]
            negative_contexts = contexts[negative_indices]
            
            for p in positive_contexts:
                for n in negative_contexts:
                    self.data_pairs.append((completion, p, n))

    def get_columns(self):
        return ['completion', 'context', 'EM']

    def __getitem__(self, idx):
        return self.data_pairs[idx]
    
    def __len__(self):
        return len(self.data_pairs)