import os, sys
from typing import Any, List, Mapping
from absl import logging
import torch as pt
import torch.nn.functional as F

from config import get_cfg
from dbgs_learner import DBGSLearner
from utils import get_T_repetition

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)


class Trainer():

    def __init__(self,
                 cfg: Mapping[str, Any]) -> None:

        """
        Train dbgsl

        Args:
            
        """

        self.cfg = cfg
        self.cfg.T_repetition = get_T_repetition(cfg)
        self.model = DBGSLearner(cfg)
        if os.path.exists(cfg.state_dict_path):
            self.model.load_state_dict(pt.load(cfg.state_dict_path, map_location=f'{cfg.device}'))
        self.optimizer = pt.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def train_step(self):

        """
        Take a single optimization step on a single batch
        """

        self.optimizer.zero_grad()
        x, target = pt.rand(self.cfg.batch_size, self.cfg.n_neurons, self.cfg.T), pt.rand(self.cfg.batch_size, 1)
        out = self.model(x)
        loss = F.binary_cross_entropy(out, target)
        loss.backward()
        self.optimizer.step()

        return loss

    def post_train(self, losses: List) -> None:
        pass
        # pt.save(self.model.state_dict(), self.cfg.state_dict_path)
        # pickle.dump(losses, open(f'{self.cfg.train_result_path}', 'wb'))

    def train(self) -> Mapping[str, Any]:

        self.model.train()
        losses = []

        for t in range(self.cfg.n_episodes):

            loss = self.train_step()
            losses.append(loss.item())
            
            if t % 50 == 0:
                logging.info(f'iteration: {t} --- loss: {loss.item()}')

        self.post_train(losses)