import os
import pickle

from gen_raw_graph import *

from datasets import load_dataset


NAME_TO_SPLIT = {"chemblpre": "chembl_pretraining", "chempcba": "pcba", "chemhiv": "hiv"}

def get_chem_dataset(name):
    cache_dir = os.path.join(os.path.dirname(__file__), "../cache_data/dataset")
    data = load_dataset("haitengzhao/molecule_property_instruction", cache_dir=cache_dir, split=NAME_TO_SPLIT[name], )
    return data

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
    #TODO
    return graphs, label_lst