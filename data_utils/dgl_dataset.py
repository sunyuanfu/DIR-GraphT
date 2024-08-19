
import dgl
import torch
from torch.utils.data import Dataset as TorchDataset

# convert PyG dataset to DGL dataset


class CustomDGLDataset(TorchDataset):
    def __init__(self, name, pyg_data):
        self.name = name
        self.pyg_data = pyg_data

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        data = self.pyg_data
        g = dgl.DGLGraph()
        if self.name == 'ogbn-arxiv' or self.name == 'ogbn-products':
            edge_index = data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        else:
            edge_index = data.edge_index
        g.add_nodes(data.num_nodes)
        g.add_edges(edge_index[0], edge_index[1])

        if data.edge_attr is not None:
            g.edata['feat'] = torch.FloatTensor(data.edge_attr)
        if self.name == 'ogbn-arxiv' or self.name == 'ogbn-products':
            g = dgl.to_bidirected(g)
            print(
                f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
            g = g.remove_self_loop().add_self_loop()
            print(f"Total edges after adding self-loop {g.number_of_edges()}")
        if data.x is not None:
            g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y)
        return g

    @property
    def train_mask(self):
        return self.pyg_data.train_mask

    @property
    def val_mask(self):
        return self.pyg_data.val_mask

    @property
    def test_mask(self):
        return self.pyg_data.test_mask

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# Creat torch dataset for GraphT
class TDataset(TorchDataset):
    def __init__(self, all_subgraphs, shortest_distances, features, labels, node_mask, name="others"):
        self.all_subgraphs = all_subgraphs
        self.shortest_distances = shortest_distances
        self.features = features
        self.labels = labels
        self.node_mask = node_mask
        self.node_ids = torch.nonzero(node_mask).flatten()
        self.name = name

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx):
        node_id = self.node_ids[idx].item()
        subgraph = self.all_subgraphs[node_id]
        neighbors = subgraph['neighbors']
        mask = subgraph['mask'].bool()
        distance_matrix = self.shortest_distances[node_id]

        feature_dim = self.features.shape[1]
        neighbor_features = []
        
        if self.name == "ogbg-pcba":
            neighbor_features.extend(subgraph["subgraph_features"].numpy())
            for neighbor_id in neighbors:
                if neighbor_id == -1:
                    neighbor_features.append(torch.rand(feature_dim))
        else:
            for neighbor_id in neighbors:
                if neighbor_id == -1:
                    neighbor_features.append(torch.rand(feature_dim))
                else:
                    neighbor_features.append(self.features[neighbor_id])

        neighbor_features = [feature.to(self.features.device) for feature in neighbor_features]
        neighbor_features = torch.stack(neighbor_features)
        
        node_label = self.labels[node_id]
        
        sample = {
            'node_id': node_id,
            'distance_matrix': distance_matrix,
            'features': neighbor_features,
            'mask': mask,
            'label': node_label,
        }
        
        return sample


def create_datasets(data, all_subgraphs, shortest_distances, features, labels, name):
    train_mask = data.train_mask
    test_mask = data.test_mask
    val_mask = data.val_mask
    
    train_dataset = TDataset(
        all_subgraphs=all_subgraphs,
        shortest_distances=shortest_distances,
        features=features,
        labels=labels,
        node_mask=train_mask,
        name=name
    )
    
    test_dataset = TDataset(
        all_subgraphs=all_subgraphs,
        shortest_distances=shortest_distances,
        features=features,
        labels=labels,
        node_mask=test_mask,
        name=name
    )
    
    val_dataset = TDataset(
        all_subgraphs=all_subgraphs,
        shortest_distances=shortest_distances,
        features=features,
        labels=labels,
        node_mask=val_mask,
        name=name
    )
    
    return train_dataset, test_dataset, val_dataset