from torch.utils.data import Dataset
import numpy as np

class CompletionContextDataset(Dataset):
    def __init__(self, input_data, tokenizer, max_length=128):
        self.data = []

        for i, item in input_data.iterrows():
            completion = item['completion_content']
            em = np.asarray(item['EMs'])
            context_files = np.asarray(item['context_files'])

            positive_indices = np.where(em == 1)[0]
            negative_indices = np.where(em == 0)[0]

            positive_contexts = context_files[positive_indices]
            negative_contexts = context_files[negative_indices]

            for p in positive_contexts:
                for n in negative_contexts:
                    self.data.append((completion, p[0]['content'], n[0]['content']))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, ):
        completion, positive_context, negative_context = self.data[idx]

        completion_encoding = self.tokenizer(completion, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        positive_context_encoding = self.tokenizer(positive_context, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        negative_context_encoding = self.tokenizer(negative_context, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

        return completion_encoding, positive_context_encoding, negative_context_encoding