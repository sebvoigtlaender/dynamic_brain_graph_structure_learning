from typing import Optional
import torch as pt
import torch.nn.functional as F

def construct_graph(x_ebd: pt.Tensor) -> pt.Tensor:
    x_ebd = F.softmax(x_ebd, -1)
    adjacency_matrix = pt.matmul(x_ebd, pt.transpose(x_ebd, 2, 3))
    return adjacency_matrix

def get_coo(adjacency_matrix: pt.Tensor) -> pt.Tensor:
    edge_indices = (adjacency_matrix > 0).nonzero()
    edge_weights = adjacency_matrix[adjacency_matrix > 0]
    return edge_indices, edge_weights 

def get_node_features(x_split: pt.Tensor) -> pt.Tensor:
    x_split_avg = pt.mean(x_split, -1, keepdim=True)
    x_split_std = pt.std(x_split, -1, keepdim=True)
    x_split_cov = pt.matmul(x_split - x_split_avg, pt.transpose(x_split - x_split_avg, 2, 3))
    node_features = x_split_cov/pt.matmul(x_split_std, pt.transpose(x_split_std, 2, 3))
    return node_features

class RegionEmbedder(pt.nn.Module):

    """
    Core region embedder
    """

    def __init__(self,
                 cfg: int,
                 act_fn: Optional[str] = 'relu',
                 bias: Optional[bool] = True) -> None:

        super().__init__()
        self.input_layer = pt.nn.Linear(cfg.len_window, cfg.itcn_d)
        self.dilated_inception = 0
        self.output_fc = pt.nn.Sequential(pt.nn.Linear(cfg.itcn_d, cfg.itcn_d), pt.nn.ReLU(), pt.nn.Linear(cfg.itcn_d, cfg.ebd_d))
        
    def forward(self, x_split: pt.Tensor) -> pt.Tensor:
        x_split = self.input_layer(x_split)
        x_split = self.output_fc(x_split)
        return x_split

class SpatialAttention(pt.nn.Module):

    """
    Spatial attention
    """

    def __init__(self,
                 cfg: int,
                 act_fn: Optional[str] = 'relu',
                 bias: Optional[bool] = True) -> None:

        super().__init__()
        n_neurons_ebd = int(cfg.tau*cfg.n_neurons)
        self.spatial_attn = pt.nn.Sequential(pt.nn.Linear(cfg.n_neurons, n_neurons_ebd, bias=False),
                                          pt.nn.ReLU(),
                                          pt.nn.Linear(n_neurons_ebd, cfg.n_neurons, bias=False),
                                          pt.nn.Sigmoid())
        
    def forward(self, x_ebd: pt.Tensor) -> pt.Tensor:
        x_spatial_attn = pt.mean(x_ebd, -1)
        x_spatial_attn = self.spatial_attn(x_spatial_attn)
        x_spatial_attn = x_spatial_attn.unsqueeze(-1)
        return x_spatial_attn

class TemporalAttention(pt.nn.Module):

    """
    Spatial attention
    """

    def __init__(self,
                 cfg: int,
                 act_fn: Optional[str] = 'relu',
                 bias: Optional[bool] = True) -> None:

        super().__init__()
        self.cfg = cfg
        T_ebd = int(cfg.tau*cfg.T_repetition)
        self.temporal_attn = pt.nn.Sequential(pt.nn.Linear(cfg.T_repetition, T_ebd, bias=False),
                                     pt.nn.ReLU(),
                                     pt.nn.Linear(T_ebd, cfg.T_repetition, bias=False),
                                     pt.nn.Sigmoid())
        
    def forward(self, x_ebd: pt.Tensor) -> pt.Tensor:
        x_temporal_attn = x_ebd.view(self.cfg.batch_size, self.cfg.T_repetition, self.cfg.n_neurons*self.cfg.ebd_d)
        x_temporal_attn = pt.mean(x_temporal_attn, -1)
        x_temporal_attn = self.temporal_attn(x_temporal_attn)
        x_temporal_attn = x_temporal_attn.view(self.cfg.batch_size, self.cfg.T_repetition, 1, 1)
        return x_temporal_attn

class Sparsify(pt.nn.Module):

    """
    Sparsifyer
    """

    def __init__(self,
                 cfg: int) -> None:

        super().__init__()
        self.threshold = pt.nn.parameter.Parameter(pt.full((1,), -5.0))

    def forward(self, adjacency_matrix: pt.Tensor) -> pt.Tensor:
        sparse_adjacency = pt.relu(adjacency_matrix - pt.sigmoid(self.threshold))
        return sparse_adjacency