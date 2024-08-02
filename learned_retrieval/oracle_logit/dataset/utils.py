import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from learned_retrieval.oracle_logit.dataset.dataset import CompletionContextDataset

def merge_by_completion(data, grouped_data):
    ungrouped_data = pd.merge(data, grouped_data, how='inner', on=['completion_content'], suffixes=('', '_y'))
    ungrouped_data.drop(ungrouped_data.filter(regex='_y$').columns, axis=1, inplace=True)
    return ungrouped_data

def train_test_split_by_completion(data, test_size):
    groped_data = data.groupby(["completion_content"], as_index=False)
    groped_data = groped_data.agg(list)

    train_groped_data, test_groped_data = train_test_split(groped_data, test_size=test_size, random_state=None, shuffle=False)

    train_data = merge_by_completion(data, train_groped_data)
    test_data = merge_by_completion(data, test_groped_data)

    return train_data, test_data

def prepare_dataset(data, normalize_strategy):
    '''
    normalize_strategy:
        ["mean_std", "mean_std_clip", "mean_std_sigmoid", "min_max_clip"]
    '''
    train_data, test_data = train_test_split_by_completion(data, test_size=0.1)
    train_data, val_data = train_test_split_by_completion(train_data, test_size=0.2)

    train_dataset = CompletionContextDataset(train_data)
    val_dataset = CompletionContextDataset(val_data)
    test_dataset = CompletionContextDataset(test_data)

    if normalize_strategy == "mean_std": # for MSELoss
        mean_, std_ = train_dataset.mean_std_norm()
        val_dataset.mean_std_norm(do_test=True, mean_=mean_, std_=std_)
    elif normalize_strategy == "mean_std_clip":
        mean_, std_ = train_dataset.mean_std_norm(do_clip=True)
        val_dataset.mean_std_norm(do_test=True, mean_=mean_, std_=std_, do_clip=True)
    elif normalize_strategy == "mean_std_sigmoid":
        mean_, std_ = train_dataset.mean_std_norm(do_sigmoid=True)
        val_dataset.mean_std_norm(do_test=True, mean_=mean_, std_=std_, do_sigmoid=True)
    elif normalize_strategy == "min_max_clip":
        min_, max_ = train_dataset.min_max_norm()
        val_dataset.min_max_norm(do_test=True, min_=min_, max_=max_, do_clip=True)

    return train_dataset, val_dataset, test_dataset

def prepare_dataloader(train_dataset, val_dataset, test_dataset, batch_size, num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader