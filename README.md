![alt text](https://raw.githubusercontent.com/JeanMaximilienCadic/CIFAR10-Iyo/master/img/cifar.jpg)

# Fast training on CIFAR10 with Iyo

Fast asynchronous training (time in evaluation is saved) with [Iyo](http://iyo.ai/) on the CIFAR10 dataset.
248
## Prerequisites
- Python 3.6
- Iyo 19.10

## Benchmark
### Iyo Async (+25% faster)
==> Loading training data..
==> Building training model..

```
0/10 | Lr: 1.0000E-01 | Loss: None | Acc: None(93.9): 100%|███████| 391/391 [00:35<00:00, 11.10it/s]

1/10 | Lr: 1.0000E-01 | Loss: 1.39E-02 | Acc: 34.662(93.9): 100%|███████| 391/391 [00:30<00:00, 13.02it/s]

2/10 | Lr: 1.0000E-01 | Loss: 9.32E-03 | Acc: 57.098(93.9): 100%|███████| 391/391 [00:30<00:00, 13.01it/s]

3/10 | Lr: 1.0000E-01 | Loss: 7.19E-03 | Acc: 67.484(93.9): 100%|███████| 391/391 [00:29<00:00, 13.07it/s]

4/10 | Lr: 1.0000E-01 | Loss: 6.21E-03 | Acc: 72.32(93.9): 100%|███████| 391/391 [00:29<00:00, 13.07it/s]

5/10 | Lr: 1.0000E-01 | Loss: 5.71E-03 | Acc: 74.562(93.9): 100%|███████| 391/391 [00:30<00:00, 12.99it/s]

6/10 | Lr: 1.0000E-01 | Loss: 5.34E-03 | Acc: 76.192(93.9): 100%|███████| 391/391 [00:30<00:00, 13.00it/s]

7/10 | Lr: 1.0000E-01 | Loss: 5.12E-03 | Acc: 77.408(93.9): 100%|███████| 391/391 [00:30<00:00, 13.03it/s]

8/10 | Lr: 1.0000E-01 | Loss: 5.01E-03 | Acc: 77.948(93.9): 100%|███████| 391/391 [00:30<00:00, 12.98it/s]

9/10 | Lr: 1.0000E-01 | Loss: 4.91E-03 | Acc: 78.282(93.9): 100%|███████| 391/391 [00:30<00:00, 12.99it/s]
```
Total over 10 epochs: 0:05:09.459543

### Original version
==> Preparing data..
==> Building model..

```Epoch: 0
[==== 391/391 ==========>]  Step: 1s87ms | Tot: 36s626ms | Loss: 2.269 | Acc: 28.664%
[==== 100/100 ==========>]  Step: 43ms | Tot: 4s94ms | Loss: 1.610 | Acc: 40.120%
Saving..

Epoch: 1
[==== 391/391 ====>]  Step: 118ms | Tot: 35s773ms | Loss: 1.579 | Acc: 42.098%
[==== 100/100 ====>]  Step: 43ms | Tot: 4s47ms | Loss: 1.401 | Acc: 49.180%
Saving..

Epoch: 2
[==== 391/391 ====>]  Step: 89ms | Tot: 35s796ms | Loss: 1.397 | Acc: 49.568%
[==== 100/100 ====>]  Step: 47ms | Tot: 4s45ms | Loss: 1.249 | Acc: 55.580%
Saving..

Epoch: 3
[==== 391/391 ====>]  Step: 94ms | Tot: 35s29ms | Loss: 1.244 | Acc: 55.548%
[==== 100/100 ====>]  Step: 43ms | Tot: 4s135ms | Loss: 1.139 | Acc: 58.960%
Saving..

Epoch: 4
[==== 391/391 ====>]  Step: 101ms | Tot: 35s723ms | Loss: 1.148 | Acc: 59.194%
[==== 100/100 ====>]  Step: 40ms | Tot: 4s91ms | Loss: 1.034 | Acc: 63.060%
Saving..

Epoch: 5
[==== 391/391 ====>]  Step: 101ms | Tot: 35s650ms | Loss: 1.084 | Acc: 61.418%
[==== 100/100 ====>]  Step: 41ms | Tot: 4s67ms | Loss: 1.050 | Acc: 63.380%
Saving..

Epoch: 6
[==== 391/391 ====>]  Step: 99ms | Tot: 35s613ms | Loss: 1.013 | Acc: 64.106%
[==== 100/100 ====>]  Step: 36ms | Tot: 4s69ms | Loss: 0.940 | Acc: 66.960%
Saving..

Epoch: 7
[==== 391/391 ====>]  Step: 97ms | Tot: 35s583ms | Loss: 0.948 | Acc: 66.568%
[==== 100/100 ====>]  Step: 41ms | Tot: 4s65ms | Loss: 0.925 | Acc: 67.540%
Saving..

Epoch: 8
[==== 391/391 ====>]  Step: 93ms | Tot: 35s508ms | Loss: 0.909 | Acc: 68.308%  
[==== 100/100 ====>]  Step: 38ms | Tot: 4s11ms | Loss: 0.890 | Acc: 69.220%
Saving..

Epoch: 9
[==== 391/391 ====>]  Step: 106ms | Tot: 35s602ms | Loss: 0.886 | Acc: 69.102%            
[==== 100/100 ====>]  Step: 36ms | Tot: 4s98ms | Loss: 0.920 | Acc: 67.950%                                                 ```                  
```
Total over 10 epochs: 0:06:46.550089




## Setup
```
conda install -c pytorch torchvision
conda install -c jcadic iyo_core
```

## Train
Start the training with:
```
python main.py --arch MobileNetV2 --lr=0.01 --epochs 350
```

Resume the training with:
```
python main.py --resume --arch MobileNetV2 --lr=0.01 --epochs 350
```


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

