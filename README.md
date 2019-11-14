![alt text](https://raw.githubusercontent.com/JeanMaximilienCadic/CIFAR10-Iyo/master/img/cifar.jpg)

# Fast training on CIFAR10 with Iyo

I'm playing with [Iyo](http://iyo.ai/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6
- Iyo 19.10

## Setup
```
conda install -c pytorch torchvision
conda install -c jcadic iyo_core
```

## Setup
Start the training with ```python main.py --arch MobileNetV2 --lr=0.01 --epochs 350```
Resume the training with ```python main.py --resume --arch MobileNetV2 --lr=0.01 --epochs 350```


## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |

