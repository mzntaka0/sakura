import copy
from concurrent.futures import ProcessPoolExecutor, as_completed


class AsyncTrainer:
    def __init__(self, trainer, args):
        self.args = args
        self.trainer = trainer

    def run(self):
        with ProcessPoolExecutor() as e:
            fs = [e.submit(copy.deepcopy(self.trainer), args=self.args, mode=mode) for mode in [
                "train",
                "test"
            ]]
            for f in as_completed(fs):
                try:
                    assert f._exception is None
                except AssertionError:
                    print("EXCEPTION: ", f._exception)
                    return f._exception
