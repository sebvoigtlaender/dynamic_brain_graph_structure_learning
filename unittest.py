from typing import Any, Mapping, Optional
import numpy as np 
import numpy.random as rnd
import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from config import get_cfg

cfg = get_cfg()
cfg.batch_size = 5
cfg.act_fn = 'relu'
cfg.n_core_layers = 4
cfg.device = 'cpu'
cfg.output_act_fn = 'softmax'
cfg.n_neurons = 20
cfg.tau = 0.1
cfg.T = 100
cfg.len_window = 10
cfg.stride = 5
cfg.itcn_d = 5
cfg.ebd_d = 3



x = pt.rand(cfg.batch_size, cfg.n_neurons, cfg.T)
# print(x.shape)

T = (cfg.T - 2*(cfg.len_window - 1) - 1)//(cfg.stride+1)
print(T)

x_cprs = pt.stack([x[:, :, t*cfg.stride:t*cfg.stride+cfg.len_window] for t in range(T)], -1)
x_cprs = pt.transpose(x_cprs, 2, 3)
# print(x_cprs.shape)

input_layer = pt.nn.Linear(cfg.len_window, cfg.itcn_d)
x_cprs = input_layer(x_cprs)
output_fc = pt.nn.Sequential(pt.nn.Linear(cfg.itcn_d, cfg.itcn_d), pt.nn.ReLU(), pt.nn.Linear(cfg.itcn_d, cfg.ebd_d))
x_cprs = output_fc(x_cprs)
# print(x_cprs.shape)



def spatial_attention(cfg, x_cprs):
    x_spatial_attn = pt.mean(x_cprs, -1)
    x_spatial_attn = pt.transpose(x_spatial_attn, 1, 2)
    n_neurons_ebd = int(cfg.tau*cfg.n_neurons)
    output_fc = pt.nn.Sequential(pt.nn.Linear(cfg.n_neurons, n_neurons_ebd, bias=False),
                                 pt.nn.ReLU(),
                                 pt.nn.Linear(n_neurons_ebd, cfg.n_neurons, bias=False),
                                 pt.nn.Sigmoid())
    spatial_attn = output_fc(x_spatial_attn)
    spatial_attn = pt.transpose(spatial_attn.unsqueeze(-1), 1, 2)
    return spatial_attn

spatial_attn = spatial_attention(cfg, x_cprs)

x_cprs = spatial_attn * x_cprs
print(x_cprs.shape)
# b, V, T, K_E

# def temporal_attention(cfg, x_cprs):
temporal_attn = pt.transpose(x_cprs, 2, 3)
temporal_attn = temporal_attn.view(cfg.batch_size, cfg.n_neurons*cfg.ebd_d, T)
temporal_attn = pt.mean(temporal_attn, 1)
T_ebd = int(cfg.tau*T)
output_fc = pt.nn.Sequential(pt.nn.Linear(T, T_ebd, bias=False),
                             pt.nn.ReLU(),
                             pt.nn.Linear(T_ebd, T, bias=False),
                             pt.nn.Sigmoid())
temporal_attn = output_fc(temporal_attn)
temporal_attn = pt.transpose(temporal_attn.view(cfg.batch_size, 1, 1, T), 2, 3)

x_cprs = temporal_attn * x_cprs
print(x_cprs.shape)

x_cprs = F.softmax(x_cprs.reshape(cfg.batch_size, T, cfg.n_neurons, cfg.ebd_d), -1)
print(x_cprs.shape)

adjacency_matrix = pt.matmul(x_cprs, pt.transpose(x_cprs, 2, 3))

