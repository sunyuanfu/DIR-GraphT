import os
import pickle

from gen_raw_graph import *

from datasets import load_dataset
import torch_geometric as pyg
import torch
from scipy.sparse import csr_array
from ogb.graphproppred import PygGraphPropPredDataset

class PPAData:
    def __init__(self, datalist, labels, train_mask, val_mask, test_mask):
        self.datalist = datalist
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.features = torch.eye(300, dtype=torch.long)

def get_raw_text_ppa(use_text=False, seed=0):
    dataset = PygGraphPropPredDataset(name="ogbg-ppa")
    data = []
    labels = dataset.y.squeeze()
    labels = labels.numpy()
    for i, subgraph in enumerate(dataset):
        subgraph.x = torch.arange(subgraph.num_nodes, dtype=torch.long)
        subgraph.adj = csr_array((torch.ones(len(subgraph.edge_index[0])), (subgraph.edge_index[0], subgraph.edge_index[1]),),
                shape=(subgraph.num_nodes, subgraph.num_nodes), )
        data.append(subgraph)
    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(len(data)).bool()
    val_mask = torch.zeros(len(data)).bool()
    test_mask = torch.zeros(len(data)).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True

    return PPAData(data, labels, train_mask, val_mask, test_mask), None
    

