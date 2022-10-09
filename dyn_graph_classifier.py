from typing import Optional
import torch as pt
import torch_geometric as tg

from ops import *


class DynGraphClassifier(pt.nn.Module):

    """
    Dynamic graph classifier
    """

    def __init__(self,
                 cfg) -> None:

        super().__init__()
        self.gru = pt.nn.GRU(cfg.n_neurons, cfg.gcn_d, cfg.n_gru_layers, batch_first=True)
        self.gcn = tg.nn.GCNConv(cfg.gcn_d, cfg.gcn_d, bias=False)
        
    def forward(self, node_features, edge_index_batch, edge_attr_batch, batch) -> pt.Tensor:
        gru_input = pt.transpose(node_features, 1, 2).reshape(cfg.batch_size*cfg.n_neurons, cfg.T_repetition, cfg.n_neurons)
        gru_output, gru_hidden = self.gru(gru_input)
        gru_output = gru_output.reshape(cfg.batch_size*cfg.T_repetition*cfg.n_neurons, cfg.gcn_d)
        out = self.gcn(gru_output, edge_index_batch, edge_attr_batch, batch)