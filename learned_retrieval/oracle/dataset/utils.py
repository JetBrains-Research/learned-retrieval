import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from learned_retrieval.oracle.dataset.dataset import BaseCompletionContextDataset
from learned_retrieval.oracle.dataset.data_classes import DatasetsClass, DataLoadersClass

def load_data(path, limit_samples=None):
    print('>>Load data')

    with open(path) as f:
        data = pd.read_json(f, orient='records', lines=True)
        if limit_samples is not None:
            data = data.iloc[:limit_samples]
    
    return data

# def prepare_dataset(data, dataset_type, normalize_strategy=None):
#     '''
#     normalize_strategy:
#         ["mean_std", "mean_std_clip", "mean_std_sigmoid", "min_max_clip"]
#     '''
#     groped_data = data.groupby(["completion_content"], as_index=False)
#     groped_data = groped_data.agg(list)

#     train_data, test_data = train_test_split(groped_data, test_size=0.1, random_state=1, shuffle=True)
#     train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=1, shuffle=True)

#     train_dataset = BaseCompletionContextDataset.create_instance(dataset_type, train_data)
#     val_dataset = BaseCompletionContextDataset.create_instance(dataset_type, val_data)
#     test_dataset = BaseCompletionContextDataset.create_instance(dataset_type, test_data)

#     datasets = DatasetsClass(train_dataset, val_dataset, test_dataset)

#     if normalize_strategy is not None:
#         normalize_datasets(datasets, normalize_strategy)
        
#     return datasets

def prepare_dataset(data_split: dict, dataset_type: str, normalize_strategy: str | None = None, limit_samples: int = None, ):
    '''
    normalize_strategy:
        ["mean_std", "mean_std_clip", "mean_std_sigmoid", "min_max_clip"]
    '''

    train_data = load_data(data_split['train'], limit_samples)
    val_data = load_data(data_split['val'], limit_samples)
    test_data = load_data(data_split['test'], limit_samples)

    train_dataset = BaseCompletionContextDataset.create_instance(dataset_type, train_data)
    val_dataset = BaseCompletionContextDataset.create_instance(dataset_type, val_data)
    test_dataset = BaseCompletionContextDataset.create_instance(dataset_type, test_data)

    datasets = DatasetsClass(train_dataset, val_dataset, test_dataset)

    if normalize_strategy is not None:
        normalize_datasets(datasets, normalize_strategy)
        
    return datasets

def normalize_datasets(datasets: DatasetsClass, normalize_strategy):
    if normalize_strategy == "mean_std": # for MSELoss
        mean_, std_ = datasets.train.mean_std_norm()
        datasets.val.mean_std_norm(do_test=True, mean_=mean_, std_=std_)
    elif normalize_strategy == "mean_std_clip":
        mean_, std_ = datasets.train.mean_std_norm(do_clip=True)
        datasets.val.mean_std_norm(do_test=True, mean_=mean_, std_=std_, do_clip=True)
    elif normalize_strategy == "mean_std_sigmoid":
        mean_, std_ = datasets.train.mean_std_norm(do_sigmoid=True)
        datasets.val.mean_std_norm(do_test=True, mean_=mean_, std_=std_, do_sigmoid=True)
    elif normalize_strategy == "min_max_clip":
        min_, max_ = datasets.train.min_max_norm()
        datasets.val.min_max_norm(do_test=True, min_=min_, max_=max_, do_clip=True)

def prepare_dataloader(datasets: DatasetsClass, batch_size, num_workers):
    train_loader = DataLoader(datasets.train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(datasets.val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # test_loader = DataLoader(datasets.test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return DataLoadersClass(train_loader, val_loader)#, test_loader)