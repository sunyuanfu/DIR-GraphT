import torch
from torch import nn
import torch.nn.functional as F
from .module.encoder import Encoder
import numpy as np

class GraphTransformer(nn.Module):

    def __init__(self, n_layers, dim_in, dim_out, dim_hidden, dim_qk, dim_v, dim_ff, n_heads, drop_input,
                 dropout, drop_mu, last_layer_n_heads, level):
        super().__init__()
        self.tst_token = nn.Parameter(torch.zeros(1, 1, dim_in))
        self.level = level
        self.scheme = "uniform" #choices: uniform, average, degree
        self.encoder = Encoder(n_layers=n_layers, dim_in=dim_in, dim_out=dim_out, dim_hidden=dim_hidden, dim_qk=dim_qk, dim_v=dim_v, 
                         dim_ff=dim_ff, n_heads=n_heads, drop_input=drop_input, dropout=dropout, drop_mu=drop_mu, last_layer_n_heads=last_layer_n_heads,
                         level=self.level)
        self.bias_codebook = nn.Parameter(self.initialize_bias_codebook(6, dim_in))
        self.bias_mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), 
            nn.ReLU(),                                  
            nn.Linear(dim_hidden, 1)  
        )

    def create_bias_matrix(self, input_D):
        b, num_nodes, _ = input_D.size()
        bias_matrix = torch.zeros(b, num_nodes, num_nodes).to(input_D.device) 
        for batch_idx in range(b):
            distance_matrix = input_D[batch_idx]
            current_bias_matrix = torch.zeros(num_nodes, num_nodes).to(input_D.device)
            for dist in range(5): 
                mask = (distance_matrix == dist)
                bias_vectors = self.bias_codebook[dist].expand(num_nodes, num_nodes, -1)  # shape: [num_nodes, num_nodes, dim]
                bias_values = self.bias_mlp(bias_vectors).squeeze(-1)  # shape: [num_nodes, num_nodes]
                current_bias_matrix[mask] = bias_values[mask]
                
            mask = (distance_matrix == 9999)
            bias_vectors = self.bias_codebook[5].expand(num_nodes, num_nodes, -1)  # shape: [num_nodes, num_nodes, dim]
            bias_values = self.bias_mlp(bias_vectors).squeeze(-1)  # shape: [num_nodes, num_nodes]
            current_bias_matrix[mask] = bias_values[mask]
            bias_matrix[batch_idx] = current_bias_matrix

        return bias_matrix

    def initialize_bias_codebook(self, num_positions, d_model):

        positions = torch.arange(num_positions).float()
        angle_rates = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        angle_rads = positions.unsqueeze(1) * angle_rates

        pos_encoding = torch.zeros(num_positions, d_model)
        pos_encoding[:, 0::2] = torch.sin(angle_rads)
        
        if d_model % 2 == 0:
            pos_encoding[:, 1::2] = torch.cos(angle_rads)
        else:
            pos_encoding[:, 1::2] = torch.cos(angle_rads)[:, :-1] 

        return pos_encoding

    def forward(self, batch):

        X = batch['features']
        MASK = batch['mask']
        D_mat = batch['distance_matrix']

        b = X.size(0)
        tst_tokens = self.tst_token.expand(b, -1, -1)
        input_X = torch.cat((tst_tokens, X), dim=1)     

        num_nodes = D_mat.size(1)
        input_D = torch.full((b, num_nodes + 1, num_nodes + 1), float('inf')).to(MASK.device)
        if self.level == "node":
            update_value = torch.where(D_mat[:, 0, 0:] == 9999, 9999, D_mat[:, 0, 0:] + 1)
            input_D[:, 0, 1:] = update_value
            update_value = torch.where(D_mat[:, 0, 0:] == 9999, 9999, D_mat[:, 0, 0:] + 1)
            input_D[:, 1:, 0] = update_value
        else:
            if self.scheme == "uniform":
                input_D[:, 0, 1:] = 1 
                input_D[:, 1:, 0] = 1
            elif self.scheme == "average":
                # Compute average distance for each node (ignoring `9999` values)
                avg_distances = torch.where(D_mat == 9999, 0, D_mat).sum(dim=2) / (D_mat != 9999).sum(dim=2)

                # Normalize average distances to the range [0, 4]
                min_dist, _ = avg_distances.min(dim=1, keepdim=True)
                max_dist, _ = avg_distances.max(dim=1, keepdim=True)
                normalized_avg_distances = 4 * (avg_distances - min_dist) / (max_dist - min_dist + 1e-9)

                # Discretize the distances to integers between 0 and 4
                discretized_avg_distances = torch.round(torch.round(normalized_avg_distances).clamp(0, 4)).long()
                input_D[:, 0, 1:] = discretized_avg_distances
                input_D[:, 1:, 0] = discretized_avg_distances
            else:
                # Compute the degree (or the number of connections) for each node, excluding `9999`
                node_degrees = (D_mat != 9999).sum(dim=2)

                # Normalize node degrees to the range [0, 4]
                min_degree, _ = node_degrees.min(dim=1, keepdim=True)
                max_degree, _ = node_degrees.max(dim=1, keepdim=True)
                normalized_degrees = 4 * (node_degrees - min_degree) / (max_degree - min_degree + 1e-9)

                # Discretize the normalized degrees to integers between 0 and 4
                discretized_degrees = torch.round(torch.round(normalized_degrees).clamp(0, 4)).long()
                input_D[:, 0, 1:] = discretized_degrees
                input_D[:, 1:, 0] = discretized_degrees
        input_D[:, 1:, 1:] = D_mat
        input_D[:, :1, :1] = 0
        true_column = torch.ones((b, 1), dtype=torch.bool).to(MASK.device)
        input_MASK = torch.cat((true_column, MASK), dim=1)
        # print(input_X.shape, input_MASK.shape)

        bias_mat = self.create_bias_matrix(input_D)

        attn_score, output = self.encoder(input_X, input_MASK, bias_mat)
        # with open('matrix.txt', 'w') as f:
        #     for row in input_D[0]:
        #         f.write(' '.join(map(str, row)) + '\n')

        # with open('matrix2.txt', 'w') as f:
        #     for row in bias_mat[0]:
        #         f.write(' '.join(map(str, row)) + '\n')
        return attn_score, output


