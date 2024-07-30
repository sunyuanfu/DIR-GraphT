from scipy.sparse import csr_array
import numpy as np
import torch
from data_utils.load import load_data
from tqdm import tqdm
from torch.utils.data import DataLoader

def sample_fixed_hop_size_neighbor(adj_mat, root, hop, max_nodes_per_hop):
    visited = np.array(root)
    fringe = np.array(root)
    nodes = np.array([])
    for h in range(1, hop + 1):
        u = adj_mat[fringe].nonzero()[1]
        fringe = np.setdiff1d(u, visited)
        visited = np.union1d(visited, fringe)
        if len(fringe) > max_nodes_per_hop:
            fringe = np.random.choice(fringe, max_nodes_per_hop, replace=False)
        if len(fringe) == 0:
            break
        nodes = np.concatenate([nodes, fringe])
    nodes = np.unique(nodes)
    nodes = nodes.astype(int)
    return nodes
 
def get_neighbors(data, node_id, hop, max_nodes_per_hop):
    neighbors = sample_fixed_hop_size_neighbor(data.adj, [node_id], hop, max_nodes_per_hop)
    neighbors = np.r_[node_id, neighbors]
    edges = data.adj[neighbors, :][:, neighbors].tocoo()
    edge_index = torch.stack(
        [torch.tensor(edges.row, dtype=torch.long), torch.tensor(edges.col, dtype=torch.long), ])
    return edge_index, neighbors

def shortest_dist_sparse_mult(edge_index, num_nodes, hop=6, source=None):
    adj_mat = csr_array((torch.ones(len(edge_index[0])), (edge_index[0], edge_index[1]),),
                    shape=(num_nodes, num_nodes), )
    if source is not None:
        neighbor_adj = adj_mat[source]
        ind = source
    else:
        neighbor_adj = adj_mat
        ind = np.arange(adj_mat.shape[0])
    neighbor_adj_set = [neighbor_adj]
    neighbor_dist = neighbor_adj.todense()
    for i in range(hop - 1):
        new_adj = neighbor_adj_set[i].dot(adj_mat)
        neighbor_adj_set.append(new_adj)
        update_ind = (new_adj.sign() - np.sign(neighbor_dist)) == 1
        r, c = update_ind.nonzero()
        neighbor_dist[r, c] = i + 2
    neighbor_dist[neighbor_dist < 1] = 9999
    neighbor_dist[np.arange(len(neighbor_dist)), ind] = 0
    return np.asarray(neighbor_dist)

def pad_neighbors(all_subgraphs, max_length):
    for node_id, subgraph in all_subgraphs.items():
        neighbors = subgraph['neighbors']
        mask = subgraph['mask']
        if len(neighbors) < max_length:
            padding = torch.full((max_length - len(neighbors),), -1, dtype=torch.long)
            mask_padding = torch.zeros((max_length - len(neighbors),), dtype=torch.float32)
            neighbors = torch.cat([neighbors, padding])
            mask = torch.cat([mask, mask_padding])
        all_subgraphs[node_id]['neighbors'] = neighbors
        all_subgraphs[node_id]['mask'] = mask
    return all_subgraphs

def generate_all_subgraphs(data, hop=2, max_nodes_per_hop=100):
    all_subgraphs = {}
    num_nodes = data.num_nodes
    max_neighbors = 0

    for node_id in tqdm(range(num_nodes), desc="Generating Subgraphs"):
        edge_index, neighbors = get_neighbors(data, node_id, hop, max_nodes_per_hop)
        neighbors = torch.tensor(neighbors, dtype=torch.long)
        mask = torch.ones(len(neighbors), dtype=torch.float32)  
        all_subgraphs[node_id] = {
            'edge_index': edge_index,
            'neighbors': neighbors,
            'mask': mask
        }
        max_neighbors = max(max_neighbors, len(neighbors))

    all_subgraphs = pad_neighbors(all_subgraphs, max_neighbors)
    return all_subgraphs, max_neighbors

def compute_shortest_distances(all_subgraphs, max_neighbors, hop=6):
    shortest_distances = {}
    
    for node_id, subgraph in tqdm(all_subgraphs.items(), desc="Computing Shortest Distances"):
        edge_index = subgraph['edge_index']
        neighbors = subgraph['neighbors']
        num_nodes = len(neighbors)  # Number of nodes in the subgraph
        dist_matrix = shortest_dist_sparse_mult(edge_index, num_nodes, hop)
        shortest_distances[node_id] = dist_matrix

    return shortest_distances

def find_max_distance(distance_matrix):
    distance_matrix = np.where(distance_matrix == 9999, np.nan, distance_matrix)
    max_distance = np.nanmax(distance_matrix)
    return max_distance


# data, num_classes, text = load_data(dataset='cora', use_dgl=False, use_text=True, use_gpt=False, seed=0)
# all_subgraphs, max_neighbors = generate_all_subgraphs(data)
# # print(all_subgraphs[0]['edge_index'])
# # print(all_subgraphs[0]['neighbors'])
# # print(all_subgraphs[0]['mask'])
# shortest_distances = compute_shortest_distances(all_subgraphs, max_neighbors)

# print(shortest_distances[0])

# with open('matrix.txt', 'w') as f:
#     for row in shortest_distances[0]:
#         f.write(' '.join(map(str, row)) + '\n')
# edge_index, neighbors = get_neighbors(data=data, node_id=0, hop=2, max_nodes_per_hop=100)
# print(edge_index.shape)
# print(edge_index)
# print(neighbors.shape)
# print(neighbors)

# print(data.train_id[0:10])
# print(all_subgraphs[1]['edge_index'], all_subgraphs[2]['edge_index'])
# # print(shortest_distances[])
# LM_emb_path = f"/gpfsnyu/scratch/ys6310/GraphT/prt_lm/cora/deberta-base-seed0.emb"
# features = torch.from_numpy(np.array(
#     np.memmap(LM_emb_path, mode='r',
#                 dtype=np.float16,
#                 shape=(data.y.shape[0], 768)))
# ).to(torch.float32)

# from data_utils.dgl_dataset import TDataset, create_datasets

# train_dataset, test_dataset, val_dataset = create_datasets(
#     data=data,
#     all_subgraphs=all_subgraphs,
#     shortest_distances=shortest_distances,
#     features=features,
#     labels=data.y
# )

# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)

# # from Transformer.module.encoder import Encoder
# from Transformer.model import GraphTransformer

# grapht = GraphTransformer(n_layers=12, dim_in=768, dim_out=7, dim_hidden=256, dim_qk=256, dim_v=256, 
#                          dim_ff=256, n_heads=16, drop_input=0., dropout=0., drop_mu=0., last_layer_n_heads=16)

# for batch in train_loader:
#     # print(batch['features'].size(0), batch['features'].size(1))
#     # print(batch['mask'][0])
#     # print(batch['distance_matrix'].dtype)
#     a, b = grapht(batch)
#     break


def get_neighbors_joint_graph(data, node_id, hop, max_nodes_per_hop):
    neighbors = sample_fixed_hop_size_neighbor(data.adj, [node_id], hop, max_nodes_per_hop)
    neighbors = np.r_[node_id, neighbors]
    edges = data.adj[neighbors, :][:, neighbors].tocoo()
    edge_index = torch.stack(
        [torch.tensor(edges.row, dtype=torch.long), torch.tensor(edges.col, dtype=torch.long), ]
    )
    new_node_id = data.num_nodes
    new_edge_id = len(neighbors)
    neighbors = np.r_[neighbors, new_node_id]
    new_edges = torch.tensor([[new_edge_id, 0], [0, new_edge_id]], dtype=torch.long)
    edge_index = torch.cat([new_edges, edge_index], dim=1)
    return edge_index, neighbors