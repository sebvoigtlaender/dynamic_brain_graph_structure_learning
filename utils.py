import torch as pt

def get_T_repetition(cfg):
    T_repetition = (cfg.T - 2*(cfg.len_window - 1) - 1)//(cfg.stride+1)
    return T_repetition

def get_x_split(cfg, x):
    x_split = pt.stack([x[:, :, t*cfg.stride:t*cfg.stride+cfg.len_window] for t in range(cfg.T_repetition)], 1)
    return x_split