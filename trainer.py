'''Train CIFAR10 with PyTorch.'''
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from tqdm import tqdm
from decimal import Decimal
class Trainer:
    def __init__(self, args):
        self.args = args
        self.arch = self.args.arch
        self.model_dir = "./checkpoint/{arch}".format(arch=self.arch)
        self.epochs = self.args.epochs
        self.init_lr = self.args.lr
        self.acc = None
        self.loss = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.start_epoch = 0

        # Data
        print('==> Loading training data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128 * 5, shuffle=True, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Model
        print('==> Building training model..')
        net = eval(self.args.arch)()
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        if self.args.resume:
            # Load checkpoint.
            print('==> Resuming from best checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            try:
                checkpoint = torch.load('{model_dir}/ckpt.pth'.format(model_dir=self.model_dir))
            except:
                checkpoint = torch.load('{model_dir}/ckpt_train.pth'.format(model_dir=self.model_dir))
            net.load_state_dict(checkpoint['net'])
            self.start_epoch = checkpoint['epoch']

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        self.net = net
        self.loader = trainloader
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.run()

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.args.lr * (self.init_lr ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # Training
    def train(self, epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        self.adjust_learning_rate(epoch)
        best_acc = torch.load('{model_dir}/ckpt.pth'.format(model_dir=self.model_dir))["acc"]
        for batch_idx, (inputs, targets) in tqdm(enumerate(self.loader),
                                                 desc='{epoch}/{epochs} | Lr: {lr} | Loss: {loss} | '
                                                      'Acc: {acc}({best_acc})'.format(
                                                     epoch=epoch,
                                                     epochs=self.epochs,
                                                     lr='%.4E' % Decimal(self.optimizer.param_groups[0]["lr"]),
                                                     loss='%.2E' % Decimal(self.loss) if self.loss is not None else None,
                                                     acc=self.acc,
                                                     best_acc=best_acc),
                                                 total=len(self.loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        self.acc = 100. * correct / total
        self.loss = train_loss/ total
        # Save checkpoint.
        state = {
            'net': self.net.state_dict(),
            'acc': self.acc,
            'epoch': epoch,
        }
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(state, '{model_dir}/ckpt_train.pth'.format(model_dir=self.model_dir,
                                                              arc=self.arch))


    def run(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.train(epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epochs', '-e', default=350, help='gives the number of epochs to train the network')
    parser.add_argument('--arch', '-a', default="MobileNetV2")
    args = parser.parse_args()
    tester = Trainer(args)
    tester.run()
