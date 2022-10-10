from typing import Any, Mapping, Optional
import torch as pt
import torch.nn.functional as F

def construct_graph(x_ebd: pt.Tensor) -> pt.Tensor:
    x_ebd = F.softmax(x_ebd, -1)
    adjacency_matrix = pt.matmul(x_ebd, pt.transpose(x_ebd, 2, 3))
    return adjacency_matrix

def get_coo(adjacency_matrix: pt.Tensor) -> pt.Tensor:
    i = 0
    edge_indices = pt.nonzero(adjacency_matrix > 0, as_tuple=False).T
    edge_index_batch = pt.clone(edge_indices[1:3, :])
    for t in range(len(edge_indices[0])):
        if i < edge_indices[0][t]:
            i = i + 1
            n_nodes = max(edge_indices[1][t-1], edge_indices[2][t-1])+1
            edge_index_batch[0][t:] = edge_index_batch[0][t:] + n_nodes
            edge_index_batch[1][t:] = edge_index_batch[1][t:] + n_nodes
    edge_attr_batch = adjacency_matrix[adjacency_matrix > 0].unsqueeze(-1)
    batch = edge_indices[0]
    return edge_index_batch, edge_attr_batch, batch

def get_node_features(x_split: pt.Tensor) -> pt.Tensor:
    x_split_avg = pt.mean(x_split, -1, keepdim=True)
    x_split_std = pt.std(x_split, -1, keepdim=True)
    x_split_cov = pt.matmul(x_split - x_split_avg, pt.transpose(x_split - x_split_avg, 2, 3))
    node_features = x_split_cov/pt.matmul(x_split_std, pt.transpose(x_split_std, 2, 3))
    return node_features

class InceptionTC(pt.nn.Module):

    """
    Inception temporal convolution layer
    """

    def __init__(self,
                 cfg: Mapping[str, Any],
                 dilation: int) -> None:

        super().__init__()
        assert cfg.itcn_d%3 == 0
        self.cfg = cfg
        self.dilation = dilation
        self.t_conv_0 = pt.nn.Conv1d(cfg.itcn_d, cfg.itcn_d//3, cfg.kernel_list[0], dilation=dilation, padding=(cfg.kernel_list[0]-1)*dilation)
        self.t_conv_1 = pt.nn.Conv1d(cfg.itcn_d, cfg.itcn_d//3, cfg.kernel_list[1], dilation=dilation, padding=(cfg.kernel_list[1]-1)*dilation)
        self.t_conv_2 = pt.nn.Conv1d(cfg.itcn_d, cfg.itcn_d//3, cfg.kernel_list[2], dilation=dilation, padding=(cfg.kernel_list[2]-1)*dilation)
        self.bn = pt.nn.BatchNorm1d(cfg.itcn_d)

    def clip_end(self, x: pt.Tensor, i: int) -> pt.Tensor:
        padding = (self.cfg.kernel_list[i]-1)*self.dilation
        x = x[:, :, :-padding].contiguous()
        return x

    def forward(self, x_split: pt.Tensor) -> pt.Tensor:
        x_cat = [self.clip_end(self.t_conv_0(x_split), 0), self.clip_end(self.t_conv_1(x_split), 1), self.clip_end(self.t_conv_2(x_split), 2)]
        x_cat = pt.cat(x_cat, 1)
        x_out = pt.relu(self.bn(x_cat))
        return x_out
        
class ITCN(pt.nn.Module):

    """
    Inception TCN
    """

    def __init__(self,
                 cfg: Mapping[str, Any],
                 n_layers: int) -> None:

        super().__init__()
        self.cfg = cfg
        self.core = pt.nn.Sequential(*[InceptionTC(cfg, 2**i) for i in range(n_layers)])

    def forward(self, x_split: pt.Tensor) -> pt.Tensor:
        x_split = x_split.reshape(self.cfg.batch_size*self.cfg.n_neurons, self.cfg.itcn_d, self.cfg.t_repetition)
        x_split = self.core(x_split)
        x_split = x_split.reshape(self.cfg.batch_size, self.cfg.t_repetition, self.cfg.n_neurons, self.cfg.itcn_d)
        return x_split

class RegionEmbedder(pt.nn.Module):

    """
    Core region embedder
    """

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.input_layer = pt.nn.Linear(cfg.len_window, cfg.itcn_d)
        self.dilated_inception = ITCN(cfg, cfg.n_itcn_layers)
        self.output_fc = pt.nn.Sequential(pt.nn.Linear(cfg.itcn_d, cfg.itcn_d), pt.nn.ReLU(), pt.nn.Linear(cfg.itcn_d, cfg.ebd_d))
        
    def forward(self, x_split: pt.Tensor) -> pt.Tensor:
        x_split = self.input_layer(x_split)
        x_split = self.dilated_inception(x_split)
        x_split = self.output_fc(x_split)
        return x_split

class SpatialAttention(pt.nn.Module):

    """
    Spatial attention
    """

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

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
    Temporal attention
    """

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.cfg = cfg
        T_ebd = int(cfg.tau*cfg.t_repetition)
        self.temporal_attn = pt.nn.Sequential(pt.nn.Linear(cfg.t_repetition, T_ebd, bias=False),
                                     pt.nn.ReLU(),
                                     pt.nn.Linear(T_ebd, cfg.t_repetition, bias=False),
                                     pt.nn.Sigmoid())
        
    def forward(self, x_ebd: pt.Tensor) -> pt.Tensor:
        x_temporal_attn = x_ebd.view(self.cfg.batch_size, self.cfg.t_repetition, self.cfg.n_neurons*self.cfg.ebd_d)
        x_temporal_attn = pt.mean(x_temporal_attn, -1)
        x_temporal_attn = self.temporal_attn(x_temporal_attn)
        x_temporal_attn = x_temporal_attn.view(self.cfg.batch_size, self.cfg.t_repetition, 1, 1)
        return x_temporal_attn

class Sparsify(pt.nn.Module):

    """
    Sparsifyer
    """

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        super().__init__()
        self.threshold = pt.nn.parameter.Parameter(pt.full((1,), -5.0))

    def forward(self, adjacency_matrix: pt.Tensor) -> pt.Tensor:
        sparse_adjacency = pt.relu(adjacency_matrix - pt.sigmoid(self.threshold))
        return sparse_adjacency