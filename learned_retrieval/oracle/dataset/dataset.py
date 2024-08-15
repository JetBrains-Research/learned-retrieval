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

    def initialize_data(self, input_data: pd.DataFrame):
        raise NotImplementedError("Subclasses must implement this method")
    
    def __len__(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def create_instance(dataset_type: str, input_data: pd.DataFrame):
        if dataset_type == 'logit':
            return LogitCompletionContextDataset(input_data)
        elif dataset_type == 'em0to1':
            return EM0to1CompletionContextDataset(input_data)
        elif dataset_type == 'em_per_file':
            return EMPerFileCompletionContextDataset(input_data)
        elif dataset_type == 'pos_neg_pairs':
            return PairsCompletionContextDataset(input_data)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

# class LogitCompletionContextDataset(BaseCompletionContextDataset):
#     def initialize_data(self, input_data: pd.DataFrame):
#         for i, item in input_data.iterrows():
#             completion = item['completion_content']
#             avg_cross_entropy = np.asarray(item['avg_cross_entropy'])
#             em = np.asarray(item['EMs'])
#             contexts = np.asarray([x[0]['content'] for x in item['context_files']])

#             for em_i, avg_ce_i, context_i in zip(em, avg_cross_entropy, contexts):
#                 self.data.append((completion, context_i, avg_ce_i, em_i))

#     def get_columns(self):
#         return ['completion', 'context', 'avg_cross_entropy', 'EM']

#     def mean_std_norm(self, do_test=False, mean_=None, std_=None, do_sigmoid=False, do_clip=False):
#         if not do_test:
#             mean_ = self.data['avg_cross_entropy'].mean()
#             std_ = self.data['avg_cross_entropy'].std()
        
#         self.data['norm_cross_entropy'] = (self.data['avg_cross_entropy'] - mean_) / std_

#         if do_sigmoid:
#             self.data['norm_cross_entropy'] = sigmoid(self.data['norm_cross_entropy'])
        
#         if do_clip:
#             self.data['norm_cross_entropy'] = np.clip(self.data['norm_cross_entropy'], 0, 1)

#         return mean_, std_

#     def min_max_norm(self, do_test=False, min_=None, max_=None, do_clip=False):
#         if not do_test:
#             min_ = self.data['avg_cross_entropy'].min()
#             max_ = self.data['avg_cross_entropy'].max()

#         self.data['norm_cross_entropy'] = (self.data['avg_cross_entropy'] - min_) / (max_ - min_)

#         if do_clip:
#             self.data['norm_cross_entropy'] = np.clip(self.data['norm_cross_entropy'], 0, 1)
        
#         return min_, max_

#     def __getitem__(self, idx):
#         completion = self.data.iloc[idx]['completion']
#         context = self.data.iloc[idx]['context']
#         norm_cross_entropy = self.data.iloc[idx]['norm_cross_entropy']
#         return completion, context, norm_cross_entropy
        
#     def __len__(self):
#         return len(self.data)
    
# class EMCompletionContextDataset(BaseCompletionContextDataset):
#     def initialize_data(self, input_data: pd.DataFrame):
#         for i, item in input_data.iterrows():
#             completion = item['completion_content']
#             em = np.asarray(item['EMs'])
#             contexts = np.asarray([x[0]['content'] for x in item['context_files']])

#             for em_i, context_i in zip(em, contexts):
#                 self.data.append((completion, context_i, float(em_i)))

#     def get_columns(self):
#         return ['completion', 'context', 'EM']

#     def __getitem__(self, idx):
#         completion = self.data.iloc[idx]['completion']
#         context = self.data.iloc[idx]['context']
#         norm_cross_entropy = self.data.iloc[idx]['EM']
#         return completion, context, norm_cross_entropy
    
#     def __len__(self):
#         return len(self.data)
    
# class PairsCompletionContextDataset(BaseCompletionContextDataset):
#     def initialize_data(self, input_data: pd.DataFrame):
#         for i, item in input_data.iterrows():
#             completion = item['completion_content']
#             em = np.asarray(item['EMs'])
#             contexts = np.asarray([x[0]['content'] for x in item['context_files']])

#             for em_i, context_i in zip(em, contexts):
#                 self.data.append((completion, context_i, float(em_i)))

#             positive_indices = np.where(em == 1)[0]
#             negative_indices = np.where(em == 0)[0]

#             positive_contexts = contexts[positive_indices]
#             negative_contexts = contexts[negative_indices]
            
#             for p in positive_contexts:
#                 for n in negative_contexts:
#                     self.data_pairs.append((completion, p, n))

#     def get_columns(self):
#         return ['completion', 'context', 'EM']

#     def __getitem__(self, idx):
#         return self.data_pairs[idx]
    
#     def __len__(self):
#         return len(self.data_pairs)

class LogitCompletionContextDataset(BaseCompletionContextDataset):
    def initialize_data(self, input_data: pd.DataFrame):
        self.data = input_data[['completion_content', 'completion_line_type', 'context_content', 'avg_cross_entropy', 'EMs']]
        # self.data['context_files'] = self.data['context_files'].apply(lambda x: x[0]['content'])

        self.data.columns = ['completion', 'completion_line_type', 'context', 'avg_cross_entropy', 'EM']

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
    
class EM0to1CompletionContextDataset(BaseCompletionContextDataset):
    def initialize_data(self, input_data: pd.DataFrame):
        choose_columns = ["completion_content", "completion_filename", "completion_line", "completion_line_type"]
        
        data = input_data[choose_columns + ["context_filename", "context_content", "EMs"]]
        data_no_context = data[data["context_filename"] == ""]
        data_no_context.columns = choose_columns + ["context_filename", "context_content", "EMs_no_context"]

        ungrouped_data = pd.merge(data, data_no_context, how='inner', on=choose_columns, suffixes=('', '_y'))
        ungrouped_data.drop(ungrouped_data.filter(regex='_y$').columns, axis=1, inplace=True)

        ungrouped_data['EM0to1'] = (ungrouped_data['EMs'] > ungrouped_data['EMs_no_context']).apply(lambda x: float(x))

        self.data = ungrouped_data[["completion_content", "completion_line_type", "context_content", "EM0to1", "EM"]]
        self.data.columns = ['completion', 'completion_line_type', 'context', 'EM0to1', 'EM']

    def __getitem__(self, idx):
        completion = self.data.iloc[idx]['completion']
        context = self.data.iloc[idx]['context']
        em_0_to_1 = self.data.iloc[idx]['EM0to1']
        return completion, context, em_0_to_1
    
    def __len__(self):
        return len(self.data)

class EMPerFileCompletionContextDataset(BaseCompletionContextDataset):
    def initialize_data(self, input_data: pd.DataFrame):
        data = input_data[['completion_filename', 'completion_content', 'completion_line_type', 'context_filename', 'context_content', 'EMs']]

        group_data = data.groupby(['completion_filename', 'context_filename'], as_index=False).agg(sum)
        group_data = group_data[['completion_filename', 'context_filename', 'EMs']]
        group_data.columns = ['completion_filename', 'context_filename', 'EMs_per_file']

        group_data_no_context = group_data[group_data["context_filename"] == ""]
        group_data_no_context.columns = ['completion_filename', 'context_filename', 'EMs_per_file_no_context']

        ungrouped_data = pd.merge(group_data, group_data_no_context, how='inner', on=['completion_filename'], suffixes=('', '_y'))
        ungrouped_data.drop(ungrouped_data.filter(regex='_y$').columns, axis=1, inplace=True)

        ungrouped_data = pd.merge(data, ungrouped_data, how='inner', on=['completion_filename', 'context_filename'], suffixes=('', '_y'))
        ungrouped_data.drop(ungrouped_data.filter(regex='_y$').columns, axis=1, inplace=True)
        
        ungrouped_data['EM_per_file'] = (ungrouped_data['EMs'] - ungrouped_data['EMs_no_context']).apply(lambda x: float(x))

        self.data = ungrouped_data[["completion_content", "completion_line_type", "context_content", "EM0to1", "EM"]]
        self.data.columns = ['completion', 'completion_line_type', 'context', 'EM_per_file', 'EM']

    def __getitem__(self, idx):
        completion = self.data.iloc[idx]['completion']
        context = self.data.iloc[idx]['context']
        em_per_file = self.data.iloc[idx]['EM_per_file']
        return completion, context, em_per_file
    
    def __len__(self):
        return len(self.data)
    
class PairsCompletionContextDataset(BaseCompletionContextDataset):
    def initialize_data(self, input_data: pd.DataFrame):
        self.data = input_data[['completion_content', 'completion_line_type', 'context_content', 'EMs']]
        # self.data['context_files'] = self.data['context_files'].apply(lambda x: x[0]['content'])
        self.data['EMs'] = self.data['EMs'].astype(float)

        self.data.columns = ['completion', 'completion_line_type', 'context', 'EM']

        self.initialize_data_pairs(input_data)

    def initialize_data_pairs(self, input_data: pd.DataFrame):
        groped_data = input_data.groupby(["completion_content"], as_index=False)
        groped_data = groped_data.agg(list)
        
        for i, item in groped_data.iterrows():
            completion = item['completion_content']
            em = np.asarray(item['EMs'])
            contexts = np.asarray(item['context_content'])
            # contexts = np.asarray([x[0]['content'] for x in item['context_files']])

            positive_indices = np.where(em == 1)[0]
            negative_indices = np.where(em == 0)[0]

            positive_contexts = contexts[positive_indices]
            negative_contexts = contexts[negative_indices]
            
            for p in positive_contexts:
                for n in negative_contexts:
                    self.data_pairs.append((completion, p, n))

    def __getitem__(self, idx):
        return self.data_pairs[idx]
    
    def __len__(self):
        return len(self.data_pairs)