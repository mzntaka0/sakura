import torch

from asyncdnn.nn.modules import AsyncSaver


def synchronize(func):
    def __record_state__(trainer):
        def record(trainer):
            trainer.state["net.{mode}".format(mode=trainer.mode)] = trainer.net.state_dict()
            trainer.state["acc.{mode}".format(mode=trainer.mode)] = trainer.acc
            trainer.state["loss.{mode}".format(mode=trainer.mode)] = trainer.loss
            trainer.state["lu.{mode}".format(mode=trainer.mode)] = trainer.lu
            trainer.state["epoch.{mode}".format(mode=trainer.mode)] = trainer.epoch

        # Update the best model
        try:
            assert "acc.{mode}".format(mode=trainer.mode) in trainer.state
            if trainer.lu == trainer.epoch and not trainer.lu == trainer.start_epoch:
                record(trainer)
                async_saver = AsyncSaver(model_path=trainer.model_path)
                trainer.state = async_saver.update(trainer.state)
        # Update the first model
        except AssertionError:
            record(trainer)
            async_saver = AsyncSaver(model_path=trainer.model_path)
            trainer.state = async_saver.update(trainer.state)

    def __sync_state__(trainer):
        try:
            trainer.state = torch.load(trainer.model_path)
            trainer.start_epoch = trainer.state["epoch.{mode}".format(mode=trainer.mode)]
        except:
            trainer.state = {}
        finally:
            __record_state__(trainer)

    def __synchronize__(*args, **kwargs):
        # print("{mode}: before run".format(mode=args[0].mode))
        __sync_state__(args[0])
        func(*args, **kwargs)
        __sync_state__(args[0])
        # print("{mode}: after run".format(mode=args[0].mode))

    return __synchronize__


if __name__ == "__main__":
    asyncsaver = AsyncSaver(model_path="checkpoint/MobileNetV2/ckpt.pth")
    state = torch.load("checkpoint/MobileNetV2/ckpt.pth")
    asyncsaver.update({
        "net.test": state["net.train"]
    })
