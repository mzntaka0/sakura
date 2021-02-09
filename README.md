![alt text](img/async.png)

# asyncdnn

Fast asynchronous training (time in evaluation is saved)  on the CIFAR10 dataset.
## Overview
## ASyncDNN modules

At a granular level, synskit is a library that consists of the following components:

| Component | Description |
| ---- | --- |
| **asyncdnn** | Fast CIFAR10 package|
| **asyncdnn.decorators** | Decorators|
| **asyncdnn.nn** | Neural network module|
| **asyncdnn.nn.models** | Models Zoo|
| **asyncdnn.nn.modules** | Modules for the network|
| **asyncdnn.trainers** | Trainers |
| **asyncdnn.test** | Test the setup|

## Setup
### Local
```
python setup.py install 
```

### Docker
```
docker build . -t jcadic/asyncdnn
docker run --rm --gpus all -it jcadic/asyncdnn bash 
```

## Test the setup
```aidl
 python -m asyncdnn.test
```
```aidl
=1= TEST PASSED : asyncdnn
=1= TEST PASSED : asyncdnn.decorators
=1= TEST PASSED : asyncdnn.nn
=1= TEST PASSED : asyncdnn.nn.models
=1= TEST PASSED : asyncdnn.nn.modules
=1= TEST PASSED : asyncdnn.nn.trainers
=1= TEST PASSED : asyncdnn.test
```

## Train
Start the training with:
```
python -m asyncdnn --arch MobileNetV2 --lr=0.01 --epochs 350
```

Resume the training with:
```
python -m asyncdnn --resume --arch MobileNetV2 --lr=0.01 --epochs 350
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



## Improvements
### Asynchronous training (+25% faster)
==> Loading training data..
==> Building training model..

```
0/10 | Lr: 1.000E-01 | LU: None(None) | Loss: INF(None) | Skipped: - | Acc: 0.000(None): 100%|████████████| 79/79 [00:17<00:00,  4.52it/s]
1/10 | Lr: 1.000E-01 | LU: 0(None) | Loss: 3.00E-03(INF) | Skipped: 0 | Acc: 31.430(0.000): 100%|████████████| 79/79 [00:11<00:00,  6.81it/s]
2/10 | Lr: 1.000E-01 | LU: 1(None) | Loss: 2.51E-03(INF) | Skipped: 0 | Acc: 53.554(0.000): 100%|████████████| 79/79 [00:13<00:00,  6.07it/s]
3/10 | Lr: 1.000E-01 | LU: 2(1) | Loss: 2.21E-03(1.18E-02) | Skipped: 0 | Acc: 63.400(58.400): 100%|████████████| 79/79 [00:13<00:00,  5.98it/s]
4/10 | Lr: 1.000E-01 | LU: 3(2) | Loss: 1.99E-03(1.04E-02) | Skipped: 0 | Acc: 69.571(63.800): 100%|████████████| 79/79 [00:13<00:00,  5.82it/s]
5/10 | Lr: 1.000E-01 | LU: 4(3) | Loss: 1.82E-03(9.12E-03) | Skipped: - | Acc: 74.268(68.890): 100%|████████████| 79/79 [00:13<00:00,  5.74it/s]
6/10 | Lr: 1.000E-01 | LU: 5(4) | Loss: 1.69E-03(7.22E-03) | Skipped: 0 | Acc: 77.512(74.750): 100%|████████████| 79/79 [00:13<00:00,  5.73it/s]
7/10 | Lr: 1.000E-01 | LU: 6(5) | Loss: 1.58E-03(7.03E-03) | Skipped: 0 | Acc: 79.802(75.500): 100%|████████████| 79/79 [00:14<00:00,  5.61it/s]
8/10 | Lr: 1.000E-01 | LU: 7(6) | Loss: 1.49E-03(6.69E-03) | Skipped: 0 | Acc: 81.390(76.770): 100%|████████████| 79/79 [00:13<00:00,  5.90it/s]
9/10 | Lr: 1.000E-01 | LU: 8(6) | Loss: 1.41E-03(6.69E-03) | Skipped: 0 | Acc: 82.826(76.770): 100%|████████████| 79/79 [00:14<00:00,  5.44it/s]
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


## Contact
For any question please contact me at j.cadic@protonmail.ch
