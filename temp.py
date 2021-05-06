import torch
import torch.nn as nn


# class A(torch.nn.Module):
#     def __init__(self):
#         super(A, self).__init__()
#         self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True, dropout=0.2)
#
#
# a = A()
#
# print(a.state_dict().keys())
#
# b = {}
# for key in a.state_dict().keys():
#     if 'weight' in key:
#         w1, w2 = a.state_dict()[key].chunk(2, 1)
#         (b[key + '.w11'], b[key + '.w12'], b[key + '.w13'], b[key + '.w14']) = w1.chunk(4, 0)
#         (b[key + '.w21'], b[key + '.w22'], b[key + '.w23'], b[key + '.w24']) = w1.chunk(4, 0)
#
# c = torch.Tensor([1000])
# b['conv.weight.w11'].copy_(c)
# print(a.state_dict())

import numpy as np
import time
import torch
import torch.nn.functional as F

# a = np.random.rand(1, 1000000)
# b = np.random.rand(1, 1000000)
# c = torch.rand(1, 1000000)
# d = torch.rand(1, 1000000)
# x = torch.rand(5, 512).cuda().unsqueeze(0)
# y = torch.rand(5*5, 512).cuda().unsqueeze(1)
# support = torch.rand(2, 5, 5, 512).cuda()
# support = torch.mean(support, 2).unsqueeze(2)
# print(support.shape)
#
# # 计算tensor在cuda上的计算速度
# time_start = time.time()
# # dist2 = F.pairwise_distance(e, f, p=2)
# # time_end = time.time()
# dist2 = torch.pow(torch.pow(x - y, 2).sum(-1), 0.5)
# time_end = time.time()
# print(x.shape)
# print(dist2.shape)
# print(time_end - time_start)
# x = torch.tensor([0.0, 1.0, 2.0])
# print(x, x.shape)
# y = F.softmax(x)
# print(y, y.shape)

import numpy as np
import matplotlib.pyplot as plt

# xx, yy = np.mgrid[0: 54: 1, 0: 54: 1]
# print(xx)
# print(yy.shape)

a = np.array([1, 2, 3])
b = np.array(a)
print(b)