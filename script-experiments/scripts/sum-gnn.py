import os
import torch
import torch.nn as nn
import numpy as np
from numpy import inf
from tqdm import tqdm
import pandas as pd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(0)

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
EXPERIMENTS_DIR = os.path.normpath(SCRIPT_PATH + "/..")
CSV_PATH = os.path.join(EXPERIMENTS_DIR, "results", "sum-gnn.csv")
THRESHOLD = 1
NUM_NODES = [
    10,
    50,
    100,
    500,
    1000,
    2000,
    5000,
    10000,
    15000,
    20000,
    50000,
    100000,
    150000,
    200000,
    500000,
]
MPNN_DIM = 64
NUM_LAYERS = 3
ER_MODEL_R = 0.5
NUM_MODELS = 10


def act_fn(x):
    return torch.clamp(x, min=-THRESHOLD, max=THRESHOLD)


class BaseGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        lim = 1
        self.w_self = -2 * lim * torch.rand(input_dim, output_dim) + lim
        self.w_neigh = -2 * lim * torch.rand(input_dim, output_dim) + lim

    def forward(self, node_feats, adj_matrix):
        node_feats_self = torch.mm(node_feats, self.w_self)
        node_feats_neigh = torch.mm(torch.mm(adj_matrix, node_feats), self.w_neigh)

        next_node_feats = node_feats_self + node_feats_neigh
        return next_node_feats
    
    def to(self, device):
        self.w_self = self.w_self.to(device)
        self.w_neigh = self.w_neigh.to(device)


class BaseGNNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, act_fn=act_fn):
        super().__init__()
        self.layers = nn.ModuleList([BaseGNNLayer(input_dim, hidden_dim)])
        for i in range(num_layers - 2):
            self.layers.append(BaseGNNLayer(hidden_dim, hidden_dim))
        self.layers.append(BaseGNNLayer(hidden_dim, output_dim))
        self.act_fn = act_fn

    def forward(self, x, adj_matrix):
        for layer in self.layers:
            x = self.act_fn(layer(x, adj_matrix))
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


gnns = []
mlps = []
for i in range(NUM_MODELS):
    gnns.append(
        BaseGNNModule(
            input_dim=MPNN_DIM,
            hidden_dim=MPNN_DIM,
            output_dim=MPNN_DIM,
            num_layers=NUM_LAYERS,
            act_fn=act_fn,
        )
    )
    mlps.append(
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
with torch.no_grad():
    for graph_dim in NUM_NODES:
        for base_gnn, mlp, mpnn_idx in zip(gnns, mlps, range(NUM_MODELS)):
            # Generate 32 graphs for each such graph dimension, to keep
            # track of the proportion that is classified as 1.
            classifications = []

            base_gnn.to(device)
            mlp.to(device)

            for idx in tqdm(range(2**5), desc=f"MPNN {mpnn_idx}, {graph_dim} nodes"):
                # Generate graph to be fed to the BaseGNN.
                half_matrix = torch.bernoulli(
                    ER_MODEL_R
                    * (torch.triu(torch.ones(graph_dim, graph_dim)) - torch.eye(graph_dim))
                )
                adj_matrix = half_matrix + half_matrix.T
                initial_node_feats = torch.rand(graph_dim, MPNN_DIM)

                adj_matrix = adj_matrix.to(device)
                initial_node_feats = initial_node_feats.to(device)

                # Obtain final mean-pooled embedding vector over all graph_dim nodes.
                output = base_gnn(initial_node_feats, adj_matrix).mean(axis=0)

                # Apply MLP classifier to the resulting output.
                apply_classifier = mlp(output)

                # If smaller than 1/2, output 0, else output 1.
                if apply_classifier <= 0.5:
                    classifications.append(0)
                else:
                    classifications.append(1)

            # Calculate proportion of graphs classified as 1.
            classifications = np.array(classifications)
            proportion = (classifications == 1).mean()

            results.loc[results_index] = [
                MPNN_DIM,
                ER_MODEL_R,
                NUM_LAYERS,
                mpnn_idx,
                graph_dim,
                proportion,
            ]
            results_index += 1

        results.to_csv(CSV_PATH, index=False)
        print(results.tail(NUM_MODELS))
