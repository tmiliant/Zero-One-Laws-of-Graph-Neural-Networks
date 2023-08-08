import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from numpy import inf
from tqdm import tqdm
import pandas as pd
import time

if not torch.cuda.is_available():
    raise Exception("CUDA not available")
device = torch.device("cuda")

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
EXPERIMENTS_DIR = os.path.normpath(SCRIPT_PATH + "/..")
CSV_PATH = os.path.join(EXPERIMENTS_DIR, "results", "gatv1.csv")
THRESHOLD = 1
NUM_NODES = [
    10,
    50,
    100,
    500,
    1000,
    2000,
]
MPNN_DIM = 64
NUM_LAYERS_LIST = [1, 2, 3]
ER_MODEL_R = 0.5
NUM_MODELS = 10
SYMMETRIZE_BATCH_SIZE = 10**7
SEED = 2376


def act_fn(x):
    return torch.clamp(x, min=-THRESHOLD, max=THRESHOLD)

class BaseGNNLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        lim = 1  
        self.w_self = -2 * lim * torch.rand(input_dim, output_dim) + lim
        self.w_neigh = -2 * lim * torch.rand(input_dim, output_dim) + lim
        self.w_readout = -2 * lim * torch.rand(input_dim, output_dim) + lim

    def forward(self, node_feats, adj_matrix):
        node_feats_self = torch.mm(node_feats, self.w_self) 
        node_feats_neigh = torch.mm(torch.mm(adj_matrix, node_feats), self.w_neigh)
        node_feats_readout = torch.mm(torch.mm(torch.ones(len(adj_matrix), len(adj_matrix)),
                                           node_feats), self.w_readout)
        next_node_feats = node_feats_self + \
                        node_feats_neigh + \
                        node_feats_readout
        return next_node_feats

    def to(self, device):
        self.w_self = self.w_self.to(device)
        self.w_neigh = self.w_neigh.to(device)
        self.w_readout = self.w_readout.to(device)

class GATV1Module(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=2, output_dim=2, num_layers=2, act_fn=act_fn):
        super().__init__()
        self.layers = nn.ModuleList([GATConv(input_dim, hidden_dim)])
        for i in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim, hidden_dim))
        self.layers.append(GATConv(hidden_dim, output_dim))
        self.act_fn = act_fn

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = self.act_fn(layer(x, edge_index))
        return x
    
    def to(self, device):
        for layer in self.layers:
            layer.to(device)


class MLPModule(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers=2, act_fn=torch.tanh
    ):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.act_fn = act_fn

    def forward(self, x):
        for layer in self.layers:
            x = self.act_fn(layer(x))
        return torch.sigmoid(x)

    def to(self, device):
        self.layers.to(device)

torch.manual_seed(SEED)

gnns = {}
mlps = {}
for num_layers in NUM_LAYERS_LIST:
    gnns[num_layers] = []
    mlps[num_layers] = []
    for i in range(NUM_MODELS):
        gnns[num_layers].append(
            GATV1Module(
                input_dim=MPNN_DIM,
                hidden_dim=MPNN_DIM,
                output_dim=MPNN_DIM,
                num_layers=num_layers,
                act_fn=act_fn,
            )
        )
        mlps[num_layers].append(
            MLPModule(
                input_dim=MPNN_DIM,
                hidden_dim=100,
                output_dim=1,
                num_layers=2,
                act_fn=torch.tanh,
            )
        )

results = pd.DataFrame(
    {
        "dimension": pd.Series(dtype="int"),
        "r": pd.Series(dtype="float"),
        "num_layers": pd.Series(dtype="int"),
        "mpnn_idx": pd.Series(dtype="int"),
        "num_nodes": pd.Series(dtype="int"),
        "proportions": pd.Series(dtype="float"),
    }
)
results_index = 0


# Create plot with x-axis an increasing seq of number of graph nodes.
for graph_dim in NUM_NODES:
    print()
    print("=======================================")
    print(f"Number of nodes: {graph_dim}")
    print("=======================================")

    for num_layers in NUM_LAYERS_LIST:
        for gnn, mlp, mpnn_idx in zip(
            gnns[num_layers], mlps[num_layers], range(NUM_MODELS)
        ):
            torch.manual_seed(SEED + mpnn_idx + num_layers + graph_dim)

            classifications = []

            gnn.to(device)
            mlp.to(device)

            for idx in tqdm(
                range(2**5), desc=f"Num layers {num_layers}, MPNN {mpnn_idx}"
            ):
                adj_matrix = torch.cuda.FloatTensor(graph_dim, graph_dim).uniform_(
                    0, ER_MODEL_R
                )
                rows_per_batch = max(
                    1, min(graph_dim, math.floor(SYMMETRIZE_BATCH_SIZE / graph_dim))
                )
                for batch_start in range(0, graph_dim, rows_per_batch):
                    batch_end = min(batch_start + rows_per_batch, graph_dim)
                    rows_this_batch = batch_end - batch_start
                    idx = torch.arange(graph_dim, device=device).tile(rows_this_batch)
                    idy = torch.arange(
                        batch_start, batch_end, device=device
                    ).repeat_interleave(graph_dim)
                    triu_mask = idx < idy
                    idx = idx[triu_mask]
                    idy = idy[triu_mask]
                    adj_matrix[idx, idy] = adj_matrix[idy, idx]
                adj_matrix[
                    torch.arange(graph_dim, device=device),
                    torch.arange(graph_dim, device=device),
                ] = 1
                edge_index = adj_matrix.nonzero().t().contiguous()
                del adj_matrix
                initial_node_feats = torch.cuda.FloatTensor(
                    graph_dim, MPNN_DIM
                ).uniform_(0, 1)

                edge_index = edge_index.to(device)
                initial_node_feats = initial_node_feats.to(device)

                # Obtain final mean-pooled embedding vector over all graph_dim nodes.
                output = gnn(initial_node_feats, edge_index).mean(axis=0)

                # Apply MLP classifier to the resulting output.
                apply_classifier = mlp(output)

                # If smaller than 1/2, output 0, else output 1.
                if apply_classifier <= 0.5:
                    classifications.append(0)
                else:
                    classifications.append(1)

                del initial_node_feats
                del output
                del edge_index

            # Calculate proportion of graphs classified as 1.
            classifications = np.array(classifications)
            proportion = (classifications == 1).mean()

            results.loc[results_index] = [
                MPNN_DIM,
                ER_MODEL_R,
                num_layers,
                mpnn_idx,
                graph_dim,
                proportion,
            ]
            results_index += 1

        results.to_csv(CSV_PATH, index=False)
        print(results.tail(NUM_MODELS))
