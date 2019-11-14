from tester import Tester
from trainer import Trainer
import argparse
from iyo.core.nn.trainers.async_trainer import AsyncTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iyo asynchronous CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epochs', '-e', default=350, help='gives the number of epochs to train the network')
    parser.add_argument('--arch', '-a', default="MobileNetV2")
    args = parser.parse_args()
    async_net = AsyncTrainer(args=args, trainer=Trainer, tester=Tester)
    async_net.run()
