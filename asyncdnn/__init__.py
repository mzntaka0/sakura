import time
import datetime
import argparse
from asyncdnn.nn.trainers import AsyncTrainer, Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epochs', '-e', default=350, type=int, help='gives the number of epochs to train the network')
    parser.add_argument('--arch', '-a', default="MobileNetV2")
    args = parser.parse_args()

    t0 = time.time()
    async_net = AsyncTrainer(trainer=Trainer, args=args)
    async_net.run()
    print(datetime.timedelta(seconds=time.time()-t0))

    # print("==========================================")
    # state = torch.load('{model_dir}/ckpt.pth'.format(model_dir="checkpoint/MobileNetV2"))
    # [print(k, v) for k, v in state.items() if not k.__contains__("net")]