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
from dataloader import FewRel, get_dataloader
from utils import neg_dist


def train_one_batch(batch, class_name0, support0, support_label, query0, query_label, mymodel, task_lr, it,
                    zero_shot=False):

    N = mymodel.n_way
    if zero_shot:
        K = 0
    else:
        K = mymodel.k_shot
    support = mymodel.coder(support0)  # [N*K, 768*2]
    query1 = mymodel.coder(query0)  # [L*N, 768*2]
    # query1 = None
    class_name1 = mymodel.coder(class_name0, is_classname=True)  # [N, 768]

    class_name = mymodel(class_name1, is_classname=True)  # ->[N, 256]
    support = mymodel(support)  # ->[N*K, 256]
    logits = neg_dist(support, class_name)  # -> [N*K, N]
    _, pred = torch.max(logits, 1)

    loss_s = mymodel.loss(logits, support_label.view(-1))
    right_s = mymodel.accuracy(pred, support_label)

    return loss_s, right_s, query1, class_name1


def train_q(batch, class_name1, query1, query_label, mymodel_clone, zero_shot=False):

    N = mymodel_clone.n_way
    if zero_shot:
        K = 0
    else:
        K = mymodel_clone.k_shot
    # support = mymodel.coder(support0)  # [N*K, 768*2]
    # query1 = mymodel.coder(query0)  # [L*N, 768*2]
    # class_name1 = mymodel.coder(class_name0, is_classname=True)  # [N, 768]

    class_name = mymodel_clone(class_name1, is_classname=True)  # ->[N, 256]
    query = mymodel_clone(query1)  # ->[L*N, 256]
    logits = neg_dist(query, class_name)  # -> [L*N, N]
    _, pred = torch.max(logits, 1)

    loss_q = mymodel_clone.loss(logits, query_label.view(-1))
    right_q = mymodel_clone.accuracy(pred, query_label)

    return loss_q, right_q


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()