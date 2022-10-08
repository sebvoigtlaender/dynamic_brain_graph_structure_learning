from typing import Any, Mapping, Optional
import numpy as np 
import numpy.random as rnd
import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch_geometric as tg
import matplotlib.pyplot as plt
from config import get_cfg
from dyn_graph_learner import DynGraphLearner
from utils import get_T_repetition, get_x_split

cfg = get_cfg()
cfg.batch_size = 5
cfg.act_fn = 'relu'
cfg.n_core_layers = 4
cfg.device = 'cpu'
cfg.output_act_fn = 'softmax'
cfg.n_hidden = 7
cfg.n_gru_layer = 1
cfg.n_neurons = 200
cfg.tau = 0.1
cfg.T = 100
cfg.len_window = 10
cfg.stride = 5
cfg.itcn_d = 4
cfg.ebd_d = 3
x = pt.rand(cfg.batch_size, cfg.n_neurons, cfg.T)
T_repetition = get_T_repetition(cfg)
cfg.T_repetition = T_repetition
dyn_graph_learner = DynGraphLearner(cfg)

x_split = get_x_split(cfg, x)
node_features, edge_indices, edge_weights = dyn_graph_learner(x_split)
print(node_features.shape)
print(edge_indices)
print(edge_weights.shape)
# gru = pt.nn.GRU(cfg.n_neurons, cfg.n_hidden, cfg.n_gru_layer, batch_first=True)
# gru_output, gru_hidden = gru(pt.transpose(node_features, 1, 2).reshape(cfg.batch_size*cfg.n_neurons, T, cfg.n_neurons))
# print(gru_output.shape)

# gcnconv = tg.nn.GCNConv(3, 2)
# gcnconv(gru_output, edge_indices, edge_weights)

# print(x_split.shape)
# x_split = pt.transpose(x_split, 1, 3)
# print(x_split.shape)
# class dilated_inception(nn.Module):
#     def __init__(self, cin, cout, dilation_factor=2):
#         super(dilated_inception, self).__init__()
#         self.tconv = nn.ModuleList()
#         self.kernel_set = [2,3,6,7]
#         cout = int(cout/len(self.kernel_set))
#         for kern in self.kernel_set:
#             self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

#     def forward(self,input):
#         x = []
#         for i in range(len(self.kernel_set)):
#             x.append(self.tconv[i](input))
#         for i in range(len(self.kernel_set)):
#             x[i] = x[iin    ][...,-x[-1].size(3):]
#         x = torch.cat(x,dim=1)
#         return x

# itcn = dilated_inception(4, 4)
# x_split = itcn(x_split)
# print(x_split.shape)