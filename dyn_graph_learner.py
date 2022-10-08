from typing import Optional
import torch as pt

from ops import *
from utils import get_T_repetition, get_x_split

class DynGraphLearner(pt.nn.Module):

    """
    Core region embedder
    """

    def __init__(self,
                 cfg: int,
                 act_fn: Optional[str] = 'relu',
                 bias: Optional[bool] = True) -> None:

        super().__init__()
        self.ebd_region = RegionEmbedder(cfg)
        self.spatial_attention = SpatialAttention(cfg)
        self.temporal_attention = TemporalAttention(cfg)
        self.sparsify = Sparsify(cfg)
        
    def forward(self, x_split: pt.Tensor) -> pt.Tensor:
        node_features = get_node_features(x_split)
        x_ebd = self.ebd_region(x_split)
        x_spatial_attention = self.spatial_attention(x_ebd)
        x_ebd = x_spatial_attention * x_ebd
        x_temporal_attn = self.temporal_attention(x_ebd)
        x_ebd = x_temporal_attn * x_ebd
        adjacency_matrix = construct_graph(x_ebd)
        sparse_adjacency = self.sparsify(adjacency_matrix)
        edge_indices, edge_weights = get_coo(sparse_adjacency)
        return node_features, edge_indices, edge_weights