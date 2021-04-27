import time
import random
import argparse
import os
import json
import numpy as np
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd

from my_transformers.transformers import AdamW
from my_transformers.transformers import BertConfig, BertModel, BertTokenizer


class FewRel(data.Dataset):
    def __init__(self, file_name, file_class, N, K, L, noise_rate):
        super(FewRel, self).__init__()
        if not os.path.isfile(file_name):
            raise Exception("[ERROR] Data file doesn't exist")
        self.json_data = json.load(open(file_name, 'r'))
        self.json_class_name = json.load(open(file_class, 'r'))
        self.classes = list(self.json_data.keys())  # example: p931
        self.N, self.K, self.L = N, K, L
        self.noise_rate = noise_rate

    def __len__(self):
        return 1000000000

    def __getitem__(self, index):
        N, K, L = self.N, self.K, self.L
        class_names = random.sample(self.classes, N)
        support, support_label, query, query_label = [], [], [], []
        class_names_final = []
        for i, name in enumerate(class_names):
            class_name = self.json_class_name[name][0] + " , it means " + self.json_class_name[name][1]
            class_name = class_name.split()
            class_names_final.append(class_name)
            rel = self.json_data[name]
            samples = random.sample(rel, K+L)
            for j in range(K):
                support.append([samples[j], i])
            for j in range(K, K+L):
                query.append([samples[j], i])

        # support=random.sample(support,N*K)
        query = random.sample(query, N*L)  # 这里相当于一个shuffle
        for i in range(N*K):
            support_label.append(support[i][1])
            support[i] = support[i][0]

        # -----这个模块是加噪声用的--------------------------------
        if self.noise_rate > 0:
            other_classes = []
            for _ in self.classes:
                if _ not in class_name:
                    other_classes.append(_)
            for i in range(N*K):
                if (random.randint(1, 10) <= self.noise_rate):  # 实现一个概率为noise_rate的加噪声的过程
                    noise_name = random.sample(other_classes, 1)
                    rel = self.json_data[noise_name[0]]
                    support[i] = random.sample(rel, 1)[0]
        # -------------------------------------------------------

        for i in range(N*L):
            query_label.append(query[i][1])
            query[i] = query[i][0]
        support_label = Variable(torch.from_numpy(np.stack(support_label, 0).astype(np.int64)).long())
        query_label = Variable(torch.from_numpy(np.stack(query_label, 0).astype(np.int64)).long())
        # if torch.cuda.is_available():support_label,query_label=support_label.cuda(),query_label.cuda()
        return class_names_final, support, support_label, query, query_label


def get_dataloader(file_name, file_class, N, K, L, noise_rate):
    data_loader = data.DataLoader(
        dataset=FewRel(file_name, file_class, N, K, L, noise_rate),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    return iter(data_loader)