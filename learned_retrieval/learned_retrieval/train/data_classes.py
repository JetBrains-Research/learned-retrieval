from dataclasses import dataclass, field
from typing import List, Any, Tuple

@dataclass
class Config:
    model_name: str
    model_type: str
    dataset_type: str
    loss: str
    device: str
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_workers: int
    max_length: int
    accumulation_steps: int
    validation_steps: int
    warmup_steps: int
    normalize_strategy: str
    limit_samples: int = None