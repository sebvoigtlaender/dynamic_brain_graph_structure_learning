from typing import Any, Mapping
import torch as pt
import torch_geometric as tg

from ops import *


class DGCLayer(pt.nn.Module):

    """
    Dynamic graph classifier layer
    """

    def __init__(self,
                 cfg: Mapping[str, Any],
                 input_d: int) -> None:

        super().__init__()
        self.cfg = cfg
        self.gru = pt.nn.GRU(input_d, cfg.gcn_d, cfg.n_gru_layers, batch_first=True)
        self.gcn = tg.nn.GCNConv(cfg.gcn_d, cfg.gcn_d, bias=False)
        
    def forward(self,
                gru_input: pt.Tensor,
                edge_index_batch: pt.Tensor,
                edge_attr_batch: pt.Tensor,
                batch: pt.Tensor) -> pt.Tensor:

        gru_output, gru_hidden = self.gru(gru_input)
        gru_output = gru_output.reshape(self.cfg.batch_size*self.cfg.t_repetition*self.cfg.n_neurons, self.cfg.gcn_d)
        out = self.gcn(gru_output, edge_index_batch, edge_attr_batch)
        out = out.reshape(self.cfg.batch_size*self.cfg.n_neurons, self.cfg.t_repetition, self.cfg.gcn_d)
        return out

class DynGraphClassifier(pt.nn.Module):

    """
    Dynamic graph classifier
    """

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.cfg = cfg
        self.dyn_graph_cls = pt.nn.ModuleList([DGCLayer(cfg, cfg.n_neurons),
                                               DGCLayer(cfg, cfg.gcn_d),
                                               DGCLayer(cfg, cfg.gcn_d)])
        
    def forward(self,
                node_features: pt.Tensor,
                edge_index_batch: pt.Tensor,
                edge_attr_batch: pt.Tensor,
                batch: pt.Tensor) -> pt.Tensor:
    
        for dgc_layer in self.dyn_graph_cls:
            node_features = dgc_layer(node_features, edge_index_batch, edge_attr_batch, batch)
        return node_features