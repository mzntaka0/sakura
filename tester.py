'''Train CIFAR10 with PyTorch.'''
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from models import *
import time
import argparse

class Tester:
    def __init__(self, args):
        self.args = args
        self.best_acc = 0  # best test accuracy
        self.model_dir = "./checkpoint/{arch}".format(arch=self.args.arch)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Data
        print('==> Loading testing data..')

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Model
        print('==> Building testing model..')
        net = eval(self.args.arch)()
        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()

        checkpoint = torch.load('{model_dir}/ckpt.pth'.format(model_dir=self.model_dir))
        net.load_state_dict(checkpoint['net'])
        self.best_acc = checkpoint["acc"]
        self.start_epoch = torch.load('{model_dir}/ckpt_train.pth'.format(model_dir=self.model_dir))['epoch']

        self.net = net
        self.criterion = criterion
        self.device = device
        self.loader = testloader
        self.run()

    def test(self, epoch):
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

                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%%/%.3f%% (%d/%d)'
                #              % (test_loss / (batch_idx + 1), 100. * correct / total, self.best_acc, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            print('Saving best model in evaluation..')
            state = {
                'net': self.net.state_dict(),
                'epoch': epoch,
                'acc': acc,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, '{model_dir}/ckpt.pth'.format(model_dir=self.model_dir))
            self.best_acc = acc

    def run(self):
        last_update=None
        while self.start_epoch < self.args.epochs:
            # Restart from train point
            _last_update = os.path.getmtime('{model_dir}/ckpt_train.pth'.format(model_dir=self.model_dir))
            if not _last_update == last_update:
                last_update = _last_update
                checkpoint = torch.load('{model_dir}/ckpt_train.pth'.format(model_dir=self.model_dir))
                self.net.load_state_dict(checkpoint['net'])
                self.start_epoch = checkpoint['epoch']
                self.test(self.start_epoch)
            else:
                # print("{},{} | Waiting for a new model to be updated...".format(self.start_epoch, self.best_acc))
                time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epochs', '-e', default=350, help='gives the number of epochs to train the network')
    parser.add_argument('--arch', '-a', default="MobileNetV2")
    args = parser.parse_args()
    tester = Tester(args)
    tester.run()