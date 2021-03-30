import os
import torch
import time
from argparse import Namespace
import numpy as np
# def save_checkpoint(func):
#     def wrapper(*args, **kwargs):
#         self = args[0]
#         state_dict = self._model.state_dict()
#         state_dict._metadata.update({"state":vars(self.state)})
#         func(*args, **kwargs, state_dict=state_dict)
#
#     return wrapper

def load_checkpoint(func):
    def wrapper(*args, **kwargs):
        self  = args[0]
        while True:
            try:
                assert os.path.exists(self._checkpoint_path)
                mtime = os.path.getmtime(self._checkpoint_path)
                assert not mtime == self.state.mtime
                state_dict = torch.load(self._checkpoint_path)
                state = state_dict._metadata["state"]
                assert state["epoch"]>=self.state.epoch
                func(*args, **kwargs, state_dict=state_dict)
                self.state = Namespace(**state)
                return
            except AssertionError:
                time.sleep(1)
                print(f"{self.state.epoch} waiting")


    return wrapper

def train(func):
    def wrapper(*args, **kwargs):
        self  = args[0]
        t0 = time.time()

        func(*args, **kwargs)

        self.state.shared.metrics.accuracy.current.train = 100. * self._correct / len(self._train_loader.dataset)
        self.state.shared.metrics.loss.current.train = np.mean(self._avg_loss)
        self.state.shared.epoch.seconds_train = time.time() - t0
        try:
            assert self.state.shared.metrics.accuracy.best.train is not None
            assert self.state.shared.metrics.accuracy.best.train > self.state.shared.metrics.accuracy.current.train
        except AssertionError:
            self.state.shared.metrics.accuracy.best.train = self.state.shared.metrics.accuracy.current.train
            self.state.shared.metrics.loss.best.train = self.state.shared.metrics.loss.current.train

    return wrapper

def test(func):
    def wrapper(*args, **kwargs):
        self  = args[0]
        # Init the metrics to 0
        metrics = self.state.shared.metrics
        metrics.loss.current.test = 0

        func(*args, **kwargs)

        # Update the metrics
        metrics.loss.current.test = self._loss / len(self._test_loader.dataset)
        metrics.accuracy.current.test = 100. * self._correct / len(self._test_loader.dataset)
        try:
            assert metrics.accuracy.best.test is not None
            assert metrics.accuracy.best.test > metrics.accuracy.current.test
        except AssertionError:
            metrics.accuracy.best.test = metrics.accuracy.current.test
            metrics.loss.best.test = metrics.loss.current.test
            self.state.shared.epoch.best = self.state.shared.epoch.current + 1


    return wrapper