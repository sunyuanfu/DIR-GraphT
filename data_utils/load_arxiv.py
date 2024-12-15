from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
from scipy.sparse import csr_array
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
def get_raw_text_arxiv(use_text=False, seed=0):

    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    idx_splits = dataset.get_idx_split()
    data.train_id = idx_splits['train']
    data.val_id = idx_splits['valid']
    data.test_id = idx_splits['test']
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    # data.edge_index = data.adj_t.to_symmetric()
    data.adj = csr_array((torch.ones(len(data.edge_index[0])), (data.edge_index[0], data.edge_index[1]),),
                    shape=(data.num_nodes, data.num_nodes), )
    if not use_text:
        return data, None
    nodeidx2paperid = pd.read_csv(
        '/gpfsnyu/scratch/ys6310/vllm_bench/dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv('/gpfsnyu/scratch/ys6310/vllm_bench/dataset/ogbn_arxiv_orig/titleabs.tsv',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'])
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    return data, text
