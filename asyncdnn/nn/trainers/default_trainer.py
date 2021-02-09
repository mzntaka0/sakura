import os
import time

from asyncdnn.nn.modules import AsyncSaver


class DefaultTrainer:
    def __init__(self, args, mode="train"):
        assert mode in ["train", "test"]
        self.args = args
        self.mode = mode
        self.model_path = None
        self.net = None
        self.start_epoch = 0
        self.epochs = self.args.epochs
        self.state = None

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def run(self):
        return self.__run__()

    def __run__(self):
        assert self.net is not None
        if self.mode == "train":
            for self.epoch in range(self.start_epoch, self.epochs):
                self.train()
        elif self.mode == "test":
            condition = True
            fingerprint = None
            while condition:
                try:
                    fingerprint = os.path.getmtime(self.model_path)
                    self.state = AsyncSaver(self.model_path).get()
                    mode = "test" if "epoch.test" in self.state else "train"
                    assert self.state['epoch.{mode}'.format(mode=mode)] + 1 < self.epochs
                    self.net.load_state_dict(self.state['net.{mode}'.format(mode=mode)])
                    self.epoch = self.state['epoch.{mode}'.format(mode=mode)]
                    self.test()
                    condition = False
                except FileNotFoundError:
                    time.sleep(1)
                except Exception:
                    print("Exception: Leavning  test...")
                    return

            while True:
                try:
                    # Restart from train point
                    _fingerprint = os.path.getmtime(self.model_path)
                    if not _fingerprint == fingerprint:
                        fingerprint = _fingerprint
                        self.state = AsyncSaver(self.model_path).get()
                        assert self.state['epoch.train'] + 1 < self.epochs
                        self.net.load_state_dict(self.state['net.train'])
                        self.epoch = self.state['epoch.train']
                        self.test()
                except AssertionError:
                    print("AssertionError: Leavning  test...")
                    return
                finally:
                    time.sleep(1)
