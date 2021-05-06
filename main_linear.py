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
from utils import neg_dist


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保证底层算法的确定性，提升可复现程度


def deep_copy(model1, model2):
    name_list = []
    for name in model1.state_dict():
        name_list.append(name)

    for name in model2.state_dict():
        if name in name_list:
            model2.state_dict()[name] = model1.state_dict()[name]

    return


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)


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

        dist_metrix = neg_dist(class_name_val_list, class_name_val_list)  # [N, N]

        for i, d in enumerate(dist_metrix):
            if i == 0:
                dist_metrix_nodiag = del_tensor_ele(d, i).view((1, -1))
            else:
                dist_metrix_nodiag = torch.cat((dist_metrix_nodiag, del_tensor_ele(d, i).view((1, -1))), dim=0)

        prob_metrix = F.softmax(dist_metrix_nodiag, dim=1)  # [N, N]
        prob_metrix = prob_metrix.cpu().numpy()

        return prob_metrix


def train_model(mymodel, mymodel_clone, args, sample_class_weights, val_step=200):

    n_way_k_shot = str(args.N) + '-way-' + str(args.K) + '-shot'
    print('Start training ' + n_way_k_shot)

    cuda = torch.cuda.is_available()
    if cuda:
        mymodel = mymodel.cuda()
        mymodel_clone = mymodel_clone.cuda()

    data_loader = {}
    data_loader['train'] = get_dataloader(args, args.train, args.class_name_file, args.N, args.K, args.L, args.noise_rate, sample_class_weights=sample_class_weights)
    data_loader['val'] = get_dataloader(args, args.val, args.class_name_file, args.N, args.K, args.L, args.noise_rate, sample_class_weights=sample_class_weights, train=False)
    data_loader['test'] = get_dataloader(args, args.test, args.class_name_file, args.N, args.K, args.L, args.noise_rate, sample_class_weights=sample_class_weights, train=False)

    optim_params=[{'params': mymodel.coder.parameters(), 'lr': 5e-5}]
    optim_params.append({'params': mymodel.fc.parameters(), 'lr': args.meta_lr})
    optim_params.append({'params': mymodel.mlp.parameters(), 'lr': args.meta_lr})
    meta_optimizer = AdamW(optim_params, lr=1)

    # mymodel1_meta_opt = AdamW(mymodel.parameters(), lr=args.meta_lr)
    # mymodel2_task_opt = AdamW(mymodel.parameters(), lr=args.task_lr)

    best_acc, best_step, best_test_acc, best_test_step, best_val_loss, best_changed = 0.0, 0, 0.0, 0, 100.0, False
    iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0
    count = 0
    count_test = 0


    for it in range(args.Train_iter):
        mymodel.train()
        meta_loss, meta_right = 0.0, 0.0
        # meta_loss = []
        # meta_right = 0.0
        # torch.save(mymodel2.state_dict(), 'model_checkpoint/checkpoint.{}th.tar'.format(it))
        for batch in range(args.B):
            class_name, support, support_label, query, query_label = next(data_loader['train'])
            # [N, length], tokens:{[N*K,length]}, [1,N*K], tokens:{[N*L,length]}, [1,N*L]
            if cuda:
                support_label, query_label = support_label.cuda(), query_label.cuda()

            '''First Step'''
            loss_s, right_s, query1, class_name1 = train_one_batch(args, class_name, support, support_label, query, query_label, mymodel,
                                                  args.task_lr, it)

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

            for _ in range(5-1):
                '''2-5th Step'''
                loss_s, right_s, query1, class_name1 = train_one_batch(args, class_name, support, support_label, query,
                                                                       query_label, mymodel_clone,
                                                                    args.task_lr, it)

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
            loss_q, right_q = train_q(args, class_name, query, query_label, mymodel_clone)
            meta_loss = meta_loss + loss_q
            meta_right = meta_right + right_q

        meta_loss_avg = meta_loss / args.B
        meta_right_avg = meta_right / args.B

        # mymodel2.load_state_dict(torch.load('model_checkpoint/checkpoint.{}th.tar'.format(it)))
        # deep_copy(mymodel1, mymodel2)

        meta_optimizer.zero_grad()
        meta_loss_avg.backward()
        meta_optimizer.step()

        iter_loss += meta_loss_avg
        iter_right += meta_right_avg

        if (it + 1) % val_step == 0:
            print('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / val_step,
                                                                       100 * iter_right / val_step))
            iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0

            count += 1
            val_acc, val_loss = test_model(cuda, data_loader['val'], mymodel, mymodel_clone, args.Val_iter, args.task_lr, meta_optimizer)
            # print('[EVAL] | loss: {0:2.6f}, accuracy: {1:2.2f}%'.format(val_loss, val_acc * 100))
            print('[EVAL] | accuracy: {0:2.2f}%'.format(val_acc * 100))
            if val_acc >= best_acc:
                print('Best checkpoint!')
                count = 0
                count_test += 1
                torch.save(mymodel.state_dict(), 'model_checkpoint/checkpoint.{0}th_best_model{1}_way_{2}_shot_Lis25_isNPM_isSW.tar'.format(it+1, args.N, args.K))
                best_acc, best_step, best_val_loss, best_changed = val_acc, (it + 1), val_loss, True
                if count_test % 5 == 0:
                    test_acc, test_loss = test_model(cuda, data_loader['test'], mymodel, mymodel_clone, args.Val_iter, args.task_lr, meta_optimizer)
                    print('[TEST] | loss: {0:2.6f}, accuracy: {1:2.2f}%'.format(test_loss, test_acc * 100))

        # torch.cuda.empty_cache()
        if count > 20:
            break

    print("\n####################\n")
    print('Finish training model! Best val acc: ' + str(best_acc) + ' at step ' + str(best_step))


def test_model(cuda, data_loader, mymodel, mymodel_clone, val_iter, task_lr, meta_optimizer, zero_shot=False):

    meta_loss_final = 0.0
    accs = 0.0
    mymodel.eval()
    for it in range(val_iter):
        meta_loss = 0.0
        # mymodel.eval()
        class_name, support, support_label, query, query_label = next(data_loader)
        if cuda:
            support_label, query_label = support_label.cuda(), query_label.cuda()

        '''First Step'''
        loss_s, right_s, query1, class_name1 = train_one_batch(args, class_name, support, support_label, query, query_label, mymodel, args.task_lr, it)

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

        '''second-10th step'''
        for _ in range(10-1):
            loss_s, right_s, query1, class_name1 = train_one_batch(args, class_name, support, support_label, query,
                                                                   query_label, mymodel_clone,
                                                                   args.task_lr, it)

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
        loss_q, right_q = train_q(args, class_name, query, query_label, mymodel_clone)
        meta_loss = meta_loss + loss_q
        meta_loss_final += loss_q
        accs += right_q

        meta_optimizer.zero_grad()
        meta_loss.backward()

        if (it+1) % 200 == 0:

            print('step: {0:4} | val_loss:{1:3.6f}, val_accuracy: {2:3.2f}%'.format(it+1, meta_loss_final/(it+1), 100*accs/(it+1)))

        # torch.cuda.empty_cache()

    return accs/val_iter, meta_loss_final/val_iter


def main(args):

    print('----------------------------------------------------')
    print("{}-way-{}-shot Few-Shot Relation Classification".format(args.N, args.K))
    print("Model: {}".format(args.Model))
    print("config:", args)
    print('----------------------------------------------------')

    start_time = time.time()
    setup_seed(args.seed)
    mymodel = MyModel(args)
    mymodel_clone = MyModel_Clone(args)
    sample_class_weights = None
    if args.ITT is True:
        sample_class_weights = pre_calculate(mymodel, args)
    train_model(mymodel, mymodel_clone, args, sample_class_weights)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Model', help='Model_name', default='Linear')
    parser.add_argument('--train', help='train file', default='data/FewRel1.0/IncreProtoNet/base_train_fewrel.json')
    parser.add_argument('--val', help='val file', default='data/FewRel1.0/IncreProtoNet/novel_val_fewrel.json')
    parser.add_argument('--test', help='test file', default='data/FewRel1.0/IncreProtoNet/novel_test_fewrel.json')
    parser.add_argument('--class_name_file', help='class name file', default='data/FewRel1.0/pid2name.json')
    parser.add_argument('--support_weights_file', help='support examples prob', default='preprocess_file/support_examples_weight_IPN.npy')
    parser.add_argument('--seed', type=int, help='seed', default=15)
    parser.add_argument('--max_length', type=int, help='max length', default=40)
    parser.add_argument('--Train_iter', type=int, help='number of iters in training', default=100000)
    parser.add_argument('--Val_iter', type=int, help='number of iters in validing', default=100)
    parser.add_argument('--Test_iter', type=int, help='number of adaptation steps', default=1000)
    parser.add_argument('--B', type=int, help='batch number', default=1)
    parser.add_argument('--N', type=int, help='N way', default=5)
    parser.add_argument('--K', type=int, help='K shot', default=1)
    parser.add_argument('--L', type=int, help='number of query per class', default=5)
    parser.add_argument('--noise_rate', type=int, help='noise rate, value range 0 to 10', default=0)
    parser.add_argument('--task_lr', type=int, help='Task learning rate(里层)', default=7e-1)
    parser.add_argument('--meta_lr', type=int, help='Meta learning rate(外层)', default=1)

    parser.add_argument('--ITT', type=bool, help='Increasing Training Tasks', default=False)
    parser.add_argument('--NPM_Loss', type=bool, help='AUX Loss, N-pair-ms loss', default=True)
    parser.add_argument('--lam', type=int, help='the importance if AUX Loss', default=1)
    parser.add_argument('--SW', type=bool, help='the weights of support instances', default=False)

    args = parser.parse_args()

    main(args)