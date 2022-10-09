from typing import Optional
import torch as pt

from ops import *
from utils import get_T_repetition, get_x_split

class DynGraphLearner(pt.nn.Module):

    """
    Dynamic graph learner
    """

    def __init__(self,
                 cfg: int) -> None:

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
        edge_index_batch, edge_attr_batch, batch = get_coo(sparse_adjacency)
        return node_features, sparse_adjacency, edge_index_batch, edge_attr_batch, batch