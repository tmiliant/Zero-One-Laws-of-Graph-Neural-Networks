import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import inf
from tqdm import tqdm
import pandas as pd

if not torch.cuda.is_available():
    raise Exception("CUDA not available")
device = torch.device("cuda")

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
]
MPNN_DIM = 64
NUM_LAYERS_LIST = [3]
NUM_LAYERS_LIST = [1]
ER_MODEL_R = 0.5
NUM_MODELS = 10
MOVE_TO_GPU_TO_SYMMETRIZE_THRESHOLD = 50000
SEED = 2343


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


gnns = {}
mlps = {}
for num_layers in NUM_LAYERS_LIST:
    gnns[num_layers] = []
    mlps[num_layers] = []
    for i in range(NUM_MODELS):
        gnns[num_layers].append(
            BaseGNNModule(
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
        for base_gnn, mlp, mpnn_idx in zip(
            gnns[num_layers], mlps[num_layers], range(NUM_MODELS)
        ):
            torch.manual_seed(SEED)

            classifications = []

            base_gnn.to(device)
            mlp.to(device)

            for idx in tqdm(
                range(2**5), desc=f"Num layers {num_layers}, MPNN {mpnn_idx}"
            ):
                # adj_matrix = generate_adjacency_matrix(graph_dim, ER_MODEL_R, RANDOM_NUMBERS_SIZE)
                adj_matrix = torch.cuda.FloatTensor(graph_dim, graph_dim).uniform_(
                    0, ER_MODEL_R
                )
                adj_matrix.triu_(diagonal=1)
                if graph_dim >= MOVE_TO_GPU_TO_SYMMETRIZE_THRESHOLD:
                    adj_matrix = adj_matrix.to("cpu")
                adj_matrix = adj_matrix + adj_matrix.T + torch.eye(graph_dim)
                initial_node_feats = torch.cuda.FloatTensor(
                    graph_dim, MPNN_DIM
                ).uniform_(0, 1)

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

                del adj_matrix

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
