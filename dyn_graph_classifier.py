from typing import Optional
import torch as pt
import torch_geometric as tg

from ops import *
from utils import get_T_repetition, get_x_split

class DynGraphClassifier(pt.nn.Module):

    """
    Dynamic graph classifier
    """

    def __init__(self,
                 cfg: int) -> None:

        super().__init__()
        self.gru = pt.nn.GRU(cfg.n_neurons, cfg.n_hidden, cfg.n_gru_layers, batch_first=True)
        self.gcnconv = tg.nn.GCNConv(3, 2)
        gru_output, gru_hidden = gru(pt.transpose(node_features, 1, 2).reshape(cfg.batch_size*cfg.n_neurons, T, cfg.n_neurons))
        gcnconv(gru_output, edge_indices, edge_weights)
        
    def forward(self, x_split: pt.Tensor) -> pt.Tensor:
        pass
