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
from train import train_one_batch, train_q, zero_grad


def test_model(mymodel, mymodel_clone, args):

    n_way_k_shot = str(args.N) + '-way-' + str(args.K) + '-shot'
    print('Start validating ' + n_way_k_shot)

    cuda = torch.cuda.is_available()
    if cuda:
        mymodel = mymodel.cuda()
        mymodel_clone = mymodel_clone.cuda()

    data_loader = {}
    # data_loader['train'] = get_dataloader(args.train, args.class_name_file, args.N, args.K, args.L, args.noise_rate)
    data_loader['val'] = get_dataloader(args.val, args.class_name_file, args.N, args.K, args.L, args.noise_rate)
    # data_loader['test'] = get_dataloader(args.test, args.class_name_file, args.N, args.K, args.L, args.noise_rate)

    optim_params = [{'params': mymodel.coder.parameters(), 'lr': 1e-8}]
    optim_params.append({'params': mymodel.fc.parameters(), 'lr': 1e-8})
    optim_params.append({'params': mymodel.mlp.parameters(), 'lr': 1e-8})
    meta_optimizer = AdamW(optim_params, lr=1)

    meta_loss_final = 0.0
    accs=0.0
    val_iter = 1000
    for it in range(val_iter):
        meta_loss = 0.0
        mymodel.eval()
        class_name, support, support_label, query, query_label = next(data_loader['val'])
        if cuda:
            support_label, query_label = support_label.cuda(), query_label.cuda()

        '''First Step'''
        loss_s, right_s, query1, class_name1 = train_one_batch(0, class_name, support, support_label, query, query_label, mymodel,
                                              args.Test_update_step, args.task_lr, it)

        zero_grad(mymodel.parameters())
        grads_fc = autograd.grad(loss_s, mymodel.fc.parameters(), retain_graph=True)
        grads_mlp = autograd.grad(loss_s, mymodel.mlp.parameters())
        fast_weights_fc, orderd_params = mymodel.cloned_fc_dict(), OrderedDict()
        fast_weights_mlp = mymodel.cloned_mlp_dict()
        for (key, val), grad in zip(mymodel.fc.named_parameters(), grads_fc):
            fast_weights_fc[key] = orderd_params['fc.' + key] = val - args.task_lr * grad
        for (key, val), grad in zip(mymodel.mlp.named_parameters(), grads_mlp):
            fast_weights_mlp[key] = orderd_params['mlp.' + key] = val - args.task_lr * grad

        name_list = []
        for name in mymodel_clone.state_dict():
            name_list.append(name)

        for name in orderd_params:
            if name in name_list:
                mymodel_clone.state_dict()[name].copy_(orderd_params[name])

        for _ in range(10-1):
            '''2-10th Step'''
            loss_s, right_s, query1, class_name1 = train_one_batch(0, class_name, support, support_label, query,
                                                                   query_label, mymodel_clone,
                                                                   args.Test_update_step, args.task_lr, it)

            zero_grad(mymodel_clone.parameters())
            grads_fc = autograd.grad(loss_s, mymodel_clone.fc.parameters(), retain_graph=True)
            grads_mlp = autograd.grad(loss_s, mymodel_clone.mlp.parameters())
            fast_weights_fc, orderd_params = mymodel_clone.cloned_fc_dict(), OrderedDict()
            fast_weights_mlp = mymodel_clone.cloned_mlp_dict()
            for (key, val), grad in zip(mymodel_clone.fc.named_parameters(), grads_fc):
                fast_weights_fc[key] = orderd_params['fc.' + key] = val - args.task_lr * grad
            for (key, val), grad in zip(mymodel_clone.mlp.named_parameters(), grads_mlp):
                fast_weights_mlp[key] = orderd_params['mlp.' + key] = val - args.task_lr * grad

            name_list = []
            for name in mymodel_clone.state_dict():
                name_list.append(name)

            for name in orderd_params:
                if name in name_list:
                    mymodel_clone.state_dict()[name].copy_(orderd_params[name])

        # -----在Query上计算loss和acc-------
        loss_q, right_q = train_q(0, class_name1, query1, query_label, mymodel_clone)
        meta_loss = meta_loss + loss_q
        meta_loss_final += loss_q
        accs += right_q

        meta_optimizer.zero_grad()
        meta_loss.backward()

        if (it+1) % 100 == 0:

            print('step: {0:4} | val_loss:{1:3.6f}, val_accuracy: {2:3.2f}%'.format(it+1, meta_loss_final/(it+1), 100*accs/(it+1)))

        torch.cuda.empty_cache()

    return accs/val_iter, meta_loss_final/val_iter


def main(args):

    print('----------------------------------------------------')
    print("{}-way-{}-shot Few-Shot Relation Classification".format(args.N, args.K))
    print("Model: {}".format(args.Model))
    print("config:", args)
    print('----------------------------------------------------')
    start_time = time.time()

    mymodel = MyModel(args)
    mymodel_clone = MyModel_Clone(args)
    mymodel.load_state_dict(torch.load('model_checkpoint/checkpoint.{}th_best_model5-way-1-shot.tar'))
    acc, loss = test_model(mymodel, mymodel_clone, args)
    print('[TEST] | loss: {0:2.6f}, accuracy: {1:2.2f}%'.format(loss, acc * 100))


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