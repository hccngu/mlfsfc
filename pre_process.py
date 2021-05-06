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
    cuda = torch.cuda.is_available()
    if cuda:
        mymodel = mymodel.cuda()

    mymodel.eval()
    with torch.no_grad():
        file_name = args.train
        file_class = args.class_name_file
        json_data = json.load(open(file_name, 'r'))
        json_class_name = json.load(open(file_class, 'r'))  # {"P931":[...], "P903":[...], ...}
        classes = list(json_data.keys())  # P931
        class_name_list = []
        class_names_final = {}
        json_class_name_prob = {}
        json_class_name_dist = {}
        for i, class_name in enumerate(json_data):
            class_name_list.append(class_name)
            val = json_class_name[class_name]
            class_name_val = val[0] + "it means " + val[1]
            class_name_val = [class_name_val.split()]
            # class_names_final[class_name] = class_name_val
            class_name_val = mymodel.coder(class_name_val, is_classname=True)  # [1, 768]
            if i == 0:
                class_name_val_list = class_name_val
            else:
                class_name_val_list = torch.cat((class_name_val_list, class_name_val), dim=0)
            # print(class_name_val.shape, type(class_name_val))
            # class_name_val_cpu = class_name_val.cpu()
            # del class_name_val
            class_names_final[class_name] = class_name_val
            # print(class_names_final[class_name].shape, type(class_names_final[class_name]))

        # class_names_val_list:[N, 768]
        for i in range(class_name_val_list.shape[0]):
            rel = json_data[class_name_list[i]]
            examples = mymodel.coder(rel)  # [700, 768*2]
            examples1 = examples[:, :768]
            examples2 = examples[:, 768:]
            examples = (examples1 + examples2)/2  # [700, 768]
            cnv = class_name_val_list[i].view((1, -1))  # [1, 768]
            dist = -neg_dist(cnv, examples)  # [1, 700]
            dist = F.softmax(dist)
            dist = dist.cpu().numpy()
            if i == 0:
                dist_list = dist
            else:
                dist_list = np.concatenate((dist_list, dist), axis=0)

        return dist_list  # [N个类，700]


def main(args):

    mymodel = MyModel(args)
    cuda = torch.cuda.is_available()
    # if cuda is True:
    #     mymodel = mymodel.cuda()
    dist_list = pre_calculate(mymodel, args)
    np.save("preprocess_file/support_examples_weight_IPN.npy", dist_list)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', help='Model_name', default='Linear')
    parser.add_argument('--train', help='train file', default='data/FewRel1.0/IncreProtoNet/base_train_fewrel.json')
    parser.add_argument('--val', help='val file', default='data/FewRel1.0/IncreProtoNet/novel_val_fewrel.json')
    parser.add_argument('--test', help='test file', default='data/FewRel1.0/IncreProtoNet/novel_test_fewrel.json')
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

    parser.add_argument('--ITT', type=bool, help='Increasing Training Tasks', default=True)
    parser.add_argument('--NPM_Loss', type=bool, help='AUX Loss, N-pair-ms loss', default=True)
    parser.add_argument('--lam', type=int, help='the importance if AUX Loss', default=0.2)
    parser.add_argument('--SW', type=bool, help='the weights of support instances', default=True)

    args = parser.parse_args()

    main(args)
    sew = np.load("preprocess_file/support_examples_weight_IPN.npy")
    print(sew.shape)
    print(type(sew))
    print(sew)