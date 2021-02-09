'''Train CIFAR10 with PyTorch'''
import torch
from torch import nn
from tqdm import tqdm
from decimal import Decimal
import multiprocessing as mp
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import datetime
import os
import argparse
from asyncdnn.nn.modules import AsyncSaver
from asyncdnn.decorators import synchronize
from asyncdnn.nn.trainers.default_trainer import DefaultTrainer
from asyncdnn.nn.models import *

class Trainer(DefaultTrainer):
    def __init__(self, args, mode="train"):
        super(Trainer, self).__init__(args=args, mode=mode)
        assert mode in ["train", "test"]
        self.args = args
        self.mode=mode
        self.arch = self.args.arch
        self.model_dir = f"__data__/pth/checkpoint/{self.arch}"
        self.model_path = f'{self.model_dir}/ckpt.pth'
        self.epochs = self.args.epochs
        self.init_lr = self.args.lr
        self.start_epoch = 0
        self.lu = None
        self.epoch = self.start_epoch
        self.acc = 0
        self.loss = np.float("inf")
        self.records = {}

        self.state = {f"net.{self.mode}": None,
                      f"epoch.{self.mode}": self.epoch,
                      f"acc.{self.mode}": self.acc,
                      f"loss.{self.mode}": self.loss,
                      f"lu.{self.mode}": self.lu}
        os.makedirs(self.model_dir, exist_ok=True)
        # Cpu/Gpu
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Data
        if self.mode=="train":
            print('==> Loading training data..')
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=10*5, shuffle=False, num_workers=mp.cpu_count())
            # Model
            print('==> Building training model..')
            self.net = eval(self.args.arch)()
            self.net = self.net.to(device)
            if device == 'cuda':
                self.net = torch.nn.DataParallel(self.net)
                cudnn.benchmark = True
        elif self.mode=="test":
            print('==> Loading testing data..')
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False,  num_workers=mp.cpu_count())
            # Model
            print('==> Building testing model..')
            self.net = eval(self.args.arch)()
            self.net = self.net.to(device)
            if device == 'cuda':
                self.net = torch.nn.DataParallel(self.net)
                cudnn.benchmark = True

        # Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.skipped=0
        self.train_loss = 0
        self.correct = 0
        self.total = 0

        # Resume
        if args.resume & os.path.exists(self.model_path):
            self.state = AsyncSaver(self.model_path).get()
            if "net.test" in self.state:
                self.net.load_state_dict(self.state['net.test'])
                self.acc = self.state["acc.test"]
                self.loss = self.state["loss.test"]
                self.start_epoch = self.state["epoch.test"]
                self.epoch = self.state["epoch.test"]
                self.lu = self.state["lu.test"]
            elif "net.{mode}".format(mode=self.mode) in self.state:
                self.net.load_state_dict(self.state['net.{mode}'.format(mode=self.mode)])
                self.acc = self.state["acc.{mode}".format(mode=self.mode)]
                self.loss = self.state["loss.{mode}".format(mode=self.mode)]
                self.start_epoch = self.state["epoch.{mode}".format(mode=self.mode)]
                self.epoch = self.state["epoch.{mode}".format(mode=self.mode)]
                self.lu = self.state["lu.{mode}".format(mode=self.mode)]
            assert self.start_epoch < self.epochs

        self.__run__()

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.args.lr * (self.init_lr ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    @synchronize
    def train(self):
        def train_batch(batch_idx, inputs, targets):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            self.train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_batch = predicted.eq(targets).sum().item()
            batch_size = targets.size(0)
            self.total += batch_size
            self.correct += correct_batch
            batch_acc = 100. * correct_batch / batch_size
            try:
                self.records[batch_idx][self.epoch % 3] = batch_acc
            except:
                self.records[batch_idx] = [batch_acc]*3
            # print(np.mean(self.records[batch_idx]))
            return batch_acc
        self.net.train()
        self.adjust_learning_rate(self.epoch)
        skipped =0
        N = len(self.loader)
        accs = []
        for batch_idx, (inputs, targets) in tqdm(enumerate(self.loader),
                                                 desc='{epoch}/{epochs} | Lr: {lr} | LU: {lu}({lu_test}) | Loss: {loss}({loss_test}) | '
                                                      'Skipped: {skipped} | Acc: {acc}({acc_test})'.format(
                                                     epoch=self.epoch,
                                                     epochs=self.epochs,
                                                     lu = self.lu,
                                                     lu_test=self.state["lu.test"]
                                                     if "lu.test" in self.state else None,
                                                     lr='%.3E' % Decimal(self.optimizer.param_groups[0]["lr"]),
                                                     loss='%.2E' % Decimal(self.loss)
                                                     if self.loss is not None else None,
                                                     loss_test='%.2E' % Decimal(self.state["loss.test"])
                                                     if "loss.test" in self.state else None,
                                                     acc='%.3F' % self.acc
                                                     if self.acc is not None else None,
                                                     acc_test='%.3F' % self.state["acc.test"]
                                                     if "acc.test" in self.state else None,
                                                     skipped=self.skipped if not self.epoch % 5 == 0 else "-"),
                                                 total=N):

            try:
                assert not self.epoch % 5 == 4
                acc_batch = np.mean(self.records[batch_idx])
                if acc_batch < 100.0:
                    batch_acc = train_batch(batch_idx, inputs, targets)
                else:
                    skipped += 1
                    batch_acc = acc_batch
            except:
                self.loader.shuffle = True
                batch_acc = train_batch(batch_idx, inputs, targets)
            finally:
                accs.append(batch_acc)

        self.lu = self.epoch
        self.skipped = skipped
        self.acc = np.mean(accs)
        self.loss = self.train_loss / self.total

    @synchronize
    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        if acc > self.acc:
            self.acc = acc
            self.lu = self.epoch
        self.loss = test_loss/total



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='gives the number of epochs to train the network')
    parser.add_argument('--arch', '-a', default="MobileNetV2")
    args = parser.parse_args()

    t0 = time.time()
    trainer = Trainer(args, mode="train")
    print(datetime.timedelta(seconds=time.time()-t0))

