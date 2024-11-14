# data_utils/load_tu_dataset.py
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

# def get_tu_dataset(name):
#     dataset = TUDataset(root=f'data/{name}', name=name)
#     num_classes = dataset.num_classes
#     return dataset, num_classes

def get_tu_dataset(name):
    # Try loading with node attributes
    dataset = TUDataset(root=f'data/{name}', name=name, use_node_attr=True)
    num_classes = dataset.num_classes

    if dataset.num_node_features == 0:
        # Node features are still missing
        print("Dataset has no node features even with use_node_attr=True.")
        print("Please consider preprocessing the dataset to generate node features.")
        raise NotImplementedError("Node features are missing and need to be generated.")

    return dataset, num_classes