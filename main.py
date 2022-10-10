from typing import Any, Mapping, Optional
from absl import logging
import torch as pt

import matplotlib.pyplot as plt

from config import get_cfg
from trainer import Trainer

def main():

    cfg = get_cfg()
    cfg.batch_size = 3
    cfg.lr = 0.003
    cfg.n_episodes = 2000
    cfg.device = 'cpu'
    cfg.n_itcn_layers = 3
    cfg.n_gru_layers = 1
    cfg.n_neurons = 10
    cfg.tau = 0.1
    cfg.T = 100
    cfg.len_window = 2
    cfg.kernel_list = [3, 5, 7]
    cfg.stride = 5
    cfg.itcn_d = 9
    cfg.ebd_d = 3
    cfg.gcn_d = 5
    cfg.n_classes = 1

    cfg.state_dict_path = ''

    logging.set_verbosity(logging.INFO)
    cfg.device = pt.device(f'cuda:{cfg.device_idx}' if pt.cuda.is_available() else 'cpu')
    trainer = Trainer(cfg)
    trainer.train()

if __name__  == '__main__':
    main()