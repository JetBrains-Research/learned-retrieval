'''
Usage: 
python3 split_logit_em_data.py  --train_file /home/kolomyttseva/Git/learned-retrieval/data/raw/train.jsonl \
                                --test_file /home/kolomyttseva/Git/learned-retrieval/data/raw/test.jsonl \
                                --split_path /home/kolomyttseva/Git/learned-retrieval/data/split
'''

from pathlib import Path
from fire import Fire
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_data(path):
    print('>> Load data')
    with open(path) as f:
        data = pd.read_json(f, orient='records', lines=True)
    return data

def merge_by_completion(data, grouped_data):
    ungrouped_data = pd.merge(data, grouped_data, how='inner', on=['completion_filename'], suffixes=('', '_y'))
    ungrouped_data.drop(ungrouped_data.filter(regex='_y$').columns, axis=1, inplace=True)
    return ungrouped_data

def train_test_split_by_completion(data, test_size):
    groped_data = data.groupby(["completion_filename"], as_index=False)
    groped_data = groped_data.agg(list)

    train_groped_data, test_groped_data = train_test_split(groped_data, test_size=test_size, random_state=1, shuffle=True)

    train_data = merge_by_completion(data, train_groped_data)
    test_data = merge_by_completion(data, test_groped_data)

    return train_data, test_data

def split_data(train_data, test_data, split_path):
    print('>> Split data')

    val, test = train_test_split_by_completion(test_data, test_size=0.3)

    train_path = Path(split_path) / 'train_split.jsonl'
    val_path = Path(split_path) / 'val_split.jsonl'
    test_path = Path(split_path) / 'test_split.jsonl'

    train_data.to_json(train_path, orient='records', lines=True)
    val.to_json(val_path, orient='records', lines=True)
    test.to_json(test_path, orient='records', lines=True)

    print(f"Train data saved to: {train_path}")
    print(f"Validation data saved to: {val_path}")
    print(f"Test data saved to: {test_path}")

# def split_data(train_data, test_data, split_path):
#     print('>> Split data')

#     train, test = train_test_split_by_completion(train_data, test_size=0.2)
#     train, val = train_test_split_by_completion(train, test_size=0.2)

#     train_path = Path(split_path) / 'train_split.jsonl'
#     val_path = Path(split_path) / 'val_split.jsonl'
#     test_path = Path(split_path) / 'test_split.jsonl'

#     train.to_json(train_path, orient='records', lines=True)
#     val.to_json(val_path, orient='records', lines=True)
#     # test_data.to_json(test_path, orient='records', lines=True)
#     test.to_json(test_path, orient='records', lines=True)

#     print(f"Train data saved to: {train_path}")
#     print(f"Validation data saved to: {val_path}")
#     print(f"Test data saved to: {test_path}")

def run(train_file: str,
        test_file: str,
        split_path: str
        ):
    
    train_data = load_data(train_file)
    # test_data = None
    test_data = load_data(test_file)
    
    # Ensure the split path directory exists
    os.makedirs(split_path, exist_ok=True)

    # Split and save the data
    split_data(train_data, test_data, split_path)

if __name__ == '__main__':
    Fire(run)