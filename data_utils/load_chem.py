import os
import pickle

from gen_raw_graph import *

from datasets import load_dataset
import torch_geometric as pyg
from scipy.sparse import csr_array
from LMs.model import SentenceEncoder


NAME_TO_SPLIT = {"chemblpre": "chembl_pretraining", "chempcba": "pcba", "chemhiv": "hiv"}

def get_chem_dataset(name):
    cache_dir = os.path.join(os.path.dirname(__file__), "../cache_data/dataset")
    data = load_dataset("haitengzhao/molecule_property_instruction", cache_dir=cache_dir, split=NAME_TO_SPLIT[name], )
    return data

class HIVData:
    def __init__(self, datalist, labels, u_node_texts_lst, train_mask, val_mask, test_mask):
        self.datalist = datalist
        self.labels = labels
        self.u_node_texts_lst = u_node_texts_lst
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.encoder = SentenceEncoder("BERT")
        self.features = self.encoder.encode(self.u_node_texts_lst)

def get_raw_text_hiv(use_text=False, seed=0):
    data = get_chem_dataset("chemhiv")
    mol = data["graph"]
    split = data["split"]
    labels = data["label"]
    label_lst = [1 if v == "Yes" else 0 for v in labels]
    graphs = []
    for i in range(len(mol)):
        graph = smiles2graph(mol[i])
        graph["label"] = label_lst[i]
        graph["split"] = split[i]
        graphs.append(graph)

    node_texts = []
    #edge_texts = []
    data = []
    for g in graphs:
        node_texts += g["node_feat"]
        #edge_texts += g["edge_feat"]
    unique_node_texts = set(node_texts)
    #unique_edge_texts = set(edge_texts)
    u_node_texts_lst = list(unique_node_texts)
    #u_edge_texts_lst = list(unique_edge_texts)
    node_texts2id = {v: i for i, v in enumerate(u_node_texts_lst)}
    #edge_texts2id = {v: i for i, v in enumerate(u_edge_texts_lst)}
    split = {"train": [], "valid": [], "test": []}
    for i, g in enumerate(graphs):
        cur_nt_id = [node_texts2id[v] for v in g["node_feat"]]
        #cur_et_id = [edge_texts2id[v] for v in g["edge_feat"]]
        subgraph = pyg.data.data.Data(x=torch.tensor(cur_nt_id, dtype=torch.long), edge_index=torch.tensor(g["edge_list"], dtype=torch.long).T,
            y=torch.tensor(g["label"]), )
        subgraph.adj = csr_array((torch.ones(len(subgraph.edge_index[0])), (subgraph.edge_index[0], subgraph.edge_index[1]),),
                shape=(subgraph.num_nodes, subgraph.num_nodes), )
        data.append(subgraph)
        split[g["split"]].append(i)

    train_mask = torch.tensor(
        [x in split["train"] for x in len(graph)])
    val_mask = torch.tensor(
        [x in split["valid"] for x in len(graph)])
    test_mask = torch.tensor(
        [x in split["test"] for x in len(graph)])

    return HIVData(data, label_lst, u_node_texts_lst, train_mask, val_mask, test_mask), None