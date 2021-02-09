import os

import torch


class AsyncSaver:
    def __init__(self, model_path):
        self.model_path = model_path
        self.state = {}

    def update(self, d):
        if os.path.exists(self.model_path):
            self.state = torch.load(self.model_path)
            for k, v in d.items():
                self.state[k] = v
        else:
            self.state = d
        torch.save(self.state, self.model_path)
        return self.state

    def get(self):
        if os.path.exists(self.model_path):
            self.state = torch.load(self.model_path)
            return self.state
