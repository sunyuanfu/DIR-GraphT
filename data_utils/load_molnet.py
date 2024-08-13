import os
import pickle

from gen_raw_graph import *

from datasets import load_dataset
import torch_geometric as pyg
from scipy.sparse import csr_array
from LMs.model import SentenceEncoder
from torch_geometric.datasets import MoleculeNet


NAME_TO_SPLIT = {"chemblpre": "chembl_pretraining", "chempcba": "pcba", "chemhiv": "hiv"}

def get_molnet_dataset(name):
    cache_dir = os.path.join(os.path.dirname(__file__), "../cache_data/dataset/MoleculeNet")
    data = MoleculeNet(root=cache_dir, name=NAME_TO_SPLIT[name])
    return data

class MolNetData:
    def __init__(self, datalist, labels, train_mask, val_mask, test_mask):
        self.datalist = datalist
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.features = torch.eye(300, dtype=torch.long)

def get_raw_text_molnet(use_text=False, seed=0):
    dataset = get_molnet_dataset("chemhiv")
    data = []
    labels = dataset.y.squeeze()
    labels = labels.numpy()
    for i, subgraph in enumerate(dataset):
        #subgraph.x = torch.arange(subgraph.num_nodes, dtype=torch.long)
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

    return MolNetData(data, labels, train_mask, val_mask, test_mask), None