from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from learned_retrieval.oracle.dataset.dataset import CompletionContextDataset

def prepare_dataset(data, tokenizer, batch_size, num_workers, max_length):
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=1, shuffle=True)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=1, shuffle=True)

    train_dataset = CompletionContextDataset(train_data, tokenizer, max_length)
    val_dataset = CompletionContextDataset(val_data, tokenizer, max_length)
    test_dataset = CompletionContextDataset(test_data, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader