from torch.utils.data import DataLoader
from dataclasses import dataclass

from learned_retrieval.oracle.dataset.dataset import BaseCompletionContextDataset

@dataclass
class DatasetsClass:
    train: BaseCompletionContextDataset = None
    val: BaseCompletionContextDataset = None
    test: BaseCompletionContextDataset = None

@dataclass
class DataLoadersClass:
    train: DataLoader = None
    val: DataLoader = None
    test: DataLoader = None