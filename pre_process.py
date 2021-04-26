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

from model import BERT, MyModel, MyModel_Clone
from utils import neg_dist


def pre_calculate(mymodel, args):
    """
    计算每个类被sample到的概率，以及类中样本被sample到的概率
    :param mymodel: model
    :param args: config
    :return: 两个列表，分别对应类概率和样本概率
    """

    mymodel.eval()
    cuda = torch.cuda.is_available()
    file_name = args.train
    file_class = args.class_name_file
    json_data = json.load(open(file_name, 'r'))
    json_class_name = json.load(open(file_class, 'r'))  # {"P931":[...], "P903":[...], ...}
    classes = list(json_data.keys())  # P931
    class_name_list = []
    class_names_final = {}
    json_class_name_prob = {}
    json_class_name_dist = {}
    for i, class_name in enumerate(json_class_name):
        class_name_list.append(class_name)
        val = json_class_name[class_name]
        class_name_val = val[0] + "it means " + val[1]
        class_name_val = [class_name_val.split()]
        # class_names_final[class_name] = class_name_val
        class_name_val = mymodel.coder(class_name_val, is_classname=True)  # [1, 768]
        # print(class_name_val.shape, type(class_name_val))
        # class_name_val_cpu = class_name_val.cpu()
        # del class_name_val
        class_names_final[class_name] = class_name_val
        # print(class_names_final[class_name].shape, type(class_names_final[class_name]))

    del mymodel
    print("class_names_final FINISH!")

    # print(class_names_final)
    length = len(class_name_list)
    C_5_id_class_name = []
    for i in range(length):
        for j in range(i+1, length):
            for k in range(j+1, length):
                for l in range(k+1, length):
                    for m in range(l+1, length):
                        C_5_id_class_name.append([i, j, k, l, m])

    print("C_5_id_class_name FINISH!")

    for i, group in enumerate(C_5_id_class_name):
        temp = []
        dist = 0.0
        for id in group:
            temp.append(class_names_final[class_name_list[id]])
        for j in range(len(temp)):
            for k in range(j+1, len(temp)):
                dist += -neg_dist(temp[j], temp[k])[0][0].detech()
        json_class_name_dist[i] = dist

    a = 1


def main(args):

    mymodel = MyModel(args)
    cuda = torch.cuda.is_available()
    # if cuda is True:
    #     mymodel = mymodel.cuda()
    pre_calculate(mymodel, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', help='Model_name', default='Linear')
    parser.add_argument('--train', help='train file', default='data/FewRel1.0/train_wiki.json')
    parser.add_argument('--val', help='val file', default='data/FewRel1.0/val_wiki.json')
    parser.add_argument('--test', help='test file', default='data/FewRel1.0/val_wiki.json')
    parser.add_argument('--class_name_file', help='class name file', default='data/FewRel1.0/pid2name.json')
    parser.add_argument('--seed', type=int, help='seed', default=15)
    parser.add_argument('--max_length', type=int, help='max length', default=30)
    parser.add_argument('--Train_iter', type=int, help='number of iters in training', default=100000)
    parser.add_argument('--Val_iter', type=int, help='number of iters in validing', default=1)
    parser.add_argument('--Test_update_step', type=int, help='number of adaptation steps', default=10)
    parser.add_argument('--B', type=int, help='batch number', default=1)
    parser.add_argument('--N', type=int, help='N way', default=5)
    parser.add_argument('--K', type=int, help='K shot', default=1)
    parser.add_argument('--L', type=int, help='number of query per class', default=25)
    parser.add_argument('--noise_rate', type=int, help='noise rate, value range 0 to 10', default=0)
    parser.add_argument('--task_lr', type=int, help='Task learning rate(里层)', default=1e-1)
    parser.add_argument('--meta_lr', type=int, help='Meta learning rate(外层)', default=1e-3)

    parser.add_argument('--ITT', type=int, help='Increasing Training Tasks', default=True)

    args = parser.parse_args()

    main(args)
