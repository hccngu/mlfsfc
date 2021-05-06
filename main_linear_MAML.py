# -*- coding: utf-8 -*-

from my_transformers.transformers import BertConfig,BertModel,BertTokenizer
from my_transformers.transformers import AdamW

import copy
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
from torch.nn import init
import torch.utils.data as data
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd


def neg_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return -torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)


class BERT(nn.Module):
    def __init__(self, N, max_length, data_dir, blank_padding=True):
        super(BERT, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.n_way = N
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.pretrained_path = 'bert-base-uncased'
        if os.path.exists(self.pretrained_path):
            config = BertConfig.from_pretrained(os.path.join(self.pretrained_path, 'bert-base-uncased-config.json'))
            self.bert = BertModel.from_pretrained(
                os.path.join(self.pretrained_path, 'bert-base-uncased-pytorch_model.bin'), config=config)
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(self.pretrained_path, 'bert-base-uncased-vocab.txt'))
        else:
            self.bert = BertModel.from_pretrained(self.pretrained_path)
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        tokens, att_masks, head_poses, outputs = [], [], [], []
        for _ in inputs:
            token, att_mask, head_pos = self.tokenize(_)
            tokens.append(token)
            att_masks.append(att_mask)
            head_poses.append(head_pos)
        token = torch.cat([t for t in tokens], 0)  # [N*K,max_length]
        att_mask = torch.cat([a for a in att_masks], 0)  # [N*K,max_length]
        # sequence_output,pooled_output=self.bert(token,attention_mask=att_mask)
        sequence_output = self.bert(token, attention_mask=att_mask)  # [N*K,max_length,bert_size]
        for i in range(token.size(0)):
            outputs.append(self.entity_start_state(head_poses[i], sequence_output[i]))
        outputs = torch.cat([o for o in outputs], 0)
        outputs = self.dropout(outputs)  # [N*K,bert_size*2]
        return outputs

    def entity_start_state(self, head_pos, sequence_output):  # 就是将BERT中两个实体前的标记位对应的输出拼接后输出作为整个句子的embedding。
        if head_pos[0] == -1 or head_pos[0] >= self.max_length:
            head_pos[0] = 0
            # raise Exception("[ERROR] no head entity")
        if head_pos[1] == -1 or head_pos[1] >= self.max_length:
            head_pos[1] = 0
            # raise Exception("[ERROR] no tail entity")
        res = torch.cat([sequence_output[head_pos[0]], sequence_output[head_pos[1]]], 0)
        return res.unsqueeze(0)

    def tokenize(self, inputs):
        tokens = inputs['tokens']
        pos_head = inputs['h'][2][0]
        pos_tail = inputs['t'][2][0]

        re_tokens, cur_pos = ['[CLS]', ], 0
        head_pos = [-1, -1]
        for token in tokens:
            token = token[0].lower()
            if cur_pos == pos_head[0]:
                head_pos[0] = len(re_tokens)
                re_tokens.append('[unused0]')
            if cur_pos == pos_tail[0]:
                head_pos[1] = len(re_tokens)
                re_tokens.append('[unused1]')
            re_tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1] - 1: re_tokens.append('[unused2]')
            if cur_pos == pos_tail[-1] - 1: re_tokens.append('[unused3]')
            cur_pos += 1
        re_tokens.append('[SEP]')

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        if self.blank_padding:
            while len(indexed_tokens) < self.max_length: indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)
        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1
        if self.cuda: indexed_tokens, att_mask = indexed_tokens.cuda(), att_mask.cuda()
        return indexed_tokens, att_mask, head_pos  # both [1,max_length]


class MyModel(nn.Module):
    def __init__(self, args, B, N, K, max_length, data_dir):
        nn.Module.__init__(self)

        self.Batch = B
        self.n_way = N
        self.k_shot = K
        self.Q = args.Q
        self.max_length = max_length
        self.data_dir = data_dir

        self.hidden_size = 768 * 2

        self.cost = nn.NLLLoss()
        self.cost2 = nn.CrossEntropyLoss()
        self.coder = BERT(N, max_length, data_dir)
        self.softmax = nn.Softmax(-1)
        self.fc = nn.Linear(self.hidden_size, 256, bias=None)
        init.kaiming_normal(self.fc.weight)

    def loss(self, logits, label, support, class_name, NPM=False, isQ=False, support_weights=None):
        if support_weights is None:
            if isQ is True:
                loss_ce = self.cost2(logits, label) / self.Q
            else:
                loss_ce = self.cost2(logits, label) / self.k_shot
        else:
            logits_softmax = F.softmax(logits, dim=-1).log()  # [N*K, N]
            support_weights_tensor = torch.from_numpy(support_weights).view(-1, 1).cuda()  # [N*K, 1]
            logits_times_weights = logits_softmax * support_weights_tensor
            loss_ce = self.nllloss(logits_times_weights, label)

        if NPM is True:
            loss_npm = torch.tensor(0.0, requires_grad=True)
            if isQ is True:
                support_N = support.view((self.n_way, self.Q, 256))
            else:
                if support_weights is not None:
                    support_weights = support_weights.reshape((self.n_way, self.k_shot))
                support_N = support.view((self.n_way, self.k_shot, 256))
            for i, s in enumerate(support_N):
                dist = -neg_dist(s, class_name) / torch.mean(-neg_dist(s, class_name), dim=0)  # [K, N]
                for j, d in enumerate(dist):
                    loss_npm_temp = torch.tensor(0.0, requires_grad=True)
                    for k, di in enumerate(d):
                        loss_npm_temp = loss_npm_temp + torch.exp(d[i] - di)
                    if support_weights is not None:
                        loss_npm = loss_npm + support_weights[i][j] * torch.log(loss_npm_temp)
                    else:
                        if isQ is True:
                            loss_npm = loss_npm + torch.log(loss_npm_temp) / self.n_way / self.Q
                        else:
                            loss_npm = loss_npm + torch.log(loss_npm_temp) / self.n_way / self.k_shot

            # print("loss_ce: ", loss_ce, "loss_npm: ", loss_npm * self.lam)
            return loss_ce + self.lam * loss_npm
        else:
            # print("loss_ce: ", loss_ce)
            return loss_ce

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def forward(self, inputs, params=None):
        if params==None:
            out = self.fc(inputs)
        else:
            out = F.linear(inputs, params['weight'])
        return F.softmax(out, dim=-1)

    def cloned_fc_dict(self):
        return {key: val.clone() for key, val in self.fc.state_dict().items()}


class FewRel(data.Dataset):
    def __init__(self, args, file_name, file_class, N, K, Q, noise_rate, sample_class_weights=None, train=True):
        super(FewRel, self).__init__()
        if not os.path.isfile(file_name):
            raise Exception("[ERROR] Data file doesn't exist")
        self.train = train
        self.json_data = json.load(open(file_name, 'r'))
        self.json_class_name = json.load(open(file_class, 'r'))
        self.classes = list(self.json_data.keys())  # example: p931
        self.N, self.K, self.Q = N, K, Q
        self.noise_rate = noise_rate
        self.sample_class_weights = sample_class_weights
        self.sew = np.load(args.support_weights_file)
        _, id_metrix = np.mgrid[0: 54: 1, 0: 54: 1]  # [54, 54]每行都是0-53
        self.id_metrix = id_metrix[~np.eye(id_metrix.shape[0], dtype=bool)].reshape(id_metrix.shape[0],
                                                                               -1)  # [54, 53]去掉了对角线元素

    def __len__(self):
        return 1000000000

    def __getitem__(self, index):
        N, K, Q = self.N, self.K, self.Q
        scw = self.sample_class_weights
        sew = self.sew
        c = np.array(self.classes)
        id_metrix = self.id_metrix
        if (scw is None) or (self.train is not True):
            class_names = random.sample(self.classes, N)
        else:
            class_names_num = []
            class_name_num = np.random.choice(len(c), 1)
            a = class_name_num[0]
            class_names_num.append(a)
            p = scw[a]
            for i in range(N-1):
                class_name_num = np.random.choice(id_metrix[a], 1, p=p, replace=False)
                a = class_name_num[0]
                class_names_num.append(a)
                p = (p + scw[a])/2

            class_names = c[class_names_num]

        support, support_label, query, query_label = [], [], [], []
        class_names_final = []
        for i, name in enumerate(class_names):
            class_name = self.json_class_name[name][0]
            class_names_final.append(class_name)
            rel = self.json_data[name]
            tem = np.argwhere(c == name)
            if self.train == True:
                samples = np.random.choice(rel, K+Q, p=sew[tem[0][0]], replace=False).tolist()
            else:
                samples = random.sample(rel, K+Q)
            for j in range(K):
                support.append([samples[j], i])
            for j in range(K, K+Q):
                query.append([samples[j], i])

        # support=random.sample(support,N*K)
        query = random.sample(query, N*Q)  # 这里相当于一个shuffle
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

        for i in range(N*Q):
            query_label.append(query[i][1])
            query[i] = query[i][0]
        support_label = Variable(torch.from_numpy(np.stack(support_label, 0).astype(np.int64)).long())
        query_label = Variable(torch.from_numpy(np.stack(query_label, 0).astype(np.int64)).long())
        # if torch.cuda.is_available():support_label,query_label=support_label.cuda(),query_label.cuda()
        return class_names_final, support, support_label, query, query_label


def get_dataloader(args, file_name, file_class, N, K, Q, noise_rate, sample_class_weights=None, train=True):
    data_loader = data.DataLoader(
        dataset=FewRel(args, file_name, file_class, N, K, Q, noise_rate, sample_class_weights, train),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    return iter(data_loader)


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def train_maml(class_name1, support1, support_label, query1, query_label, net, steps, task_lr):
    '''first step'''
    class_name, support = net(class_name1), net(support1)
    logits = neg_dist(support, class_name)
    # logits = -logits / torch.mean(logits, dim=0)
    # logits = F.softmax(logits, dim=1)
    _, pred = torch.max(logits, 1)
    loss = net.loss(logits, support_label.view(-1), support, class_name)
    zero_grad(net.parameters())
    grads = autograd.grad(loss, net.fc.parameters())
    fast_weights, orderd_params = net.cloned_fc_dict(), OrderedDict()
    for (key, val), grad in zip(net.fc.named_parameters(), grads):
        fast_weights[key] = orderd_params[key] = val-task_lr*grad
    '''steps remaining'''
    for k in range(steps-1):
        class_name, support = net(class_name1, fast_weights), net(support1, fast_weights)
        logits = neg_dist(support, class_name)
        # logits = -logits / torch.mean(logits, dim=0)
        # logits = F.softmax(logits, dim=-1)
        _, pred = torch.max(logits, 1)
        loss = net.loss(logits, support_label.view(-1), support, class_name)
        if torch.isnan(loss):
            print(loss)
        zero_grad(orderd_params.values())
        grads = torch.autograd.grad(loss, orderd_params.values())
        for (key, val), grad in zip(orderd_params.items(), grads):
            fast_weights[key] = orderd_params[key] = val-task_lr*grad
    '''return'''

    class_name = net(class_name1, fast_weights)
    query = net(query1, fast_weights)
    logits_q = neg_dist(query, class_name)
    # logits_q = -logits_q / torch.mean(logits_q, dim=0)
    # logits_q = F.softmax(logits_q, dim=1)
    _, pred = torch.max(logits_q, 1)
    loss_q = net.loss(logits_q, query_label.view(-1), query, class_name)
    return loss_q, net.accuracy(pred, query_label)


def train_one_batch(idx, class_name0, support0, support_label, query0, query_label, net, steps, task_lr, it,
                    zero_shot=False):
    print(idx)
    N = net.n_way
    if zero_shot:
        K = 0
    else:
        K = net.k_shot
    support, query, class_name = net.coder(support0), net.coder(query0), net.coder(class_name0)  # [N*K,bert_size] 这里应该是2*bert_size吧，就是将句子转为句向量

    return train_maml(class_name, support, support_label, query, query_label, net, steps, task_lr)


def test_model(cuda, data_loader, model, val_iter, test_update_step, task_lr, zero_shot=False):
    accs=0.0
    model.eval()
    for it in range(val_iter):
        net = copy.deepcopy(model)
        class_name, support, support_label, query, query_label = next(data_loader)
        if cuda:
            support_label, query_label = support_label.cuda(), query_label.cuda()
        loss, right = train_one_batch(0, class_name, support, support_label, query, query_label, net, test_update_step, task_lr, 100000000, zero_shot)
        accs+=right
        if (it+1) % 500 == 0:
            print('step: {0:4} | accuracy: {1:3.2f}%'.format(it+1, 100*accs/(it+1)))
    return accs/val_iter


def train_model(model, B, N, K, Q, data_dir,
                meta_lr=1,
                task_lr=7e-2,
                train_iter=10000,
                val_iter=1000,
                val_step=100,
                test_iter=10000,
                test_step=1000,
                test_update_step=500,
                sample_class_weights=None):
    n_way_k_shot = str(N) + '-way-' + str(K) + '-shot'
    print('Start training ' + n_way_k_shot)

    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()

    data_loader = {}
    data_loader['train'] = get_dataloader(args, args.train, args.class_name_file, args.N, args.K, args.Q,
                                          args.noise_rate, sample_class_weights=sample_class_weights)
    data_loader['val'] = get_dataloader(args, args.val, args.class_name_file, args.N, args.K, args.Q, args.noise_rate,
                                        sample_class_weights=sample_class_weights, train=False)
    data_loader['test'] = get_dataloader(args, args.test, args.class_name_file, args.N, args.K, args.Q, args.noise_rate,
                                         sample_class_weights=sample_class_weights, train=False)

    optim_params = [{'params': model.coder.parameters(), 'lr': 5e-5}]
    optim_params.append({'params': model.fc.parameters(), 'lr': args.meta_lr})
    meta_optimizer = AdamW(optim_params, lr=meta_lr)

    best_acc, best_step, best_test_acc, best_test_step, best_changed = 0.0, 0, 0.0, 0, False
    iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0

    for it in range(train_iter):
        meta_loss, meta_right = 0.0, 0.0
        for batch in range(B):
            model.train()
            class_name, support, support_label, query, query_label = next(data_loader['train'])
            # [N], [N*K,length], [1,N*K], [N*Q,length], [1,N*Q]
            if cuda:
                support_label, query_label = support_label.cuda(), query_label.cuda()

            loss, right = train_one_batch(batch, class_name, support, support_label, query, query_label, model,
                                          test_update_step, task_lr, it)
            meta_loss += loss
            meta_right += right
        meta_loss /= B
        meta_right /= B
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        iter_loss += meta_loss
        iter_right += meta_right
        iter_sample += 1

        if it % val_step == 0:
            iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0
        if ((it + 1) % 100 == 0) or ((it + 1) % val_step == 0):
            print(
            'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample,
                                                                       100 * iter_right / iter_sample))

        if (it + 1) % val_step == 0:
            acc = test_model(cuda, data_loader['val'], model, val_iter, test_update_step, task_lr)
            print('[EVAL] | accuracy: {0:2.2f}%'.format(acc * 100))
            if acc > best_acc:
                print('Best checkpoint!')
                best_model = copy.deepcopy(model)
                best_acc, best_step, best_changed = acc, (it + 1), True

        if (it + 1) % test_step == 0 and best_changed:
            best_changed = False
            test_acc = test_model(cuda, data_loader['test'], model, test_iter, test_update_step, task_lr)
            print('[TEST] | accuracy: {0:2.2f}%'.format(test_acc * 100))
            if test_acc < 1.5 / N:
                print("That's so bad!")
                break
            if test_acc > best_test_acc:
                # torch.save(best_model.state_dict(),n_way_k_shot+'.ckpt')
                best_test_acc, best_test_step = test_acc, best_step
            best_acc = 0.0

    print("\n####################\n")
    print('Finish training model! Best acc: ' + str(best_test_acc) + ' at step ' + str(best_test_step))


parser = argparse.ArgumentParser()
parser.add_argument('--train', default='data/FewRel1.0/IncreProtoNet/base_train_fewrel.json',
        help='train file')
parser.add_argument('--val', default='data/FewRel1.0/IncreProtoNet/novel_val_fewrel.json',
        help='val file')
parser.add_argument('--test', default='data/FewRel1.0/IncreProtoNet/novel_test_fewrel.json',
        help='test file')
parser.add_argument('--class_name_file', help='class name file', default='data/FewRel1.0/pid2name_final.json')
parser.add_argument('--noise_rate', default=0, type=int,
        help='noise rate, value range 0 to 10')
parser.add_argument('--B', default=2, type=int,
        help='batch number')
parser.add_argument('--N', default=5, type=int,
        help='N way')
parser.add_argument('--K', default=1, type=int,
        help='K shot')
parser.add_argument('--Q', default=5, type=int,
        help='number of query per class')
parser.add_argument('--Train_iter', default=10000, type=int,
        help='number of iters in training')
parser.add_argument('--Val_iter', default=100, type=int,
        help='number of iters in validing')
parser.add_argument('--Test_update_step', default=2, type=int,
        help='number of adaptation steps')
parser.add_argument('--max_length', default=40, type=int,
       help='max length')
parser.add_argument('--support_weights_file', help='support examples prob', default='preprocess_file/support_examples_weight_IPN.npy')
parser.add_argument('--ITT', type=bool, help='Increasing Training Tasks', default=False)
parser.add_argument('--NPM_Loss', type=bool, help='AUX Loss, N-pair-ms loss', default=False)
parser.add_argument('--lam', type=int, help='the importance if AUX Loss', default=1)
parser.add_argument('--SW', type=bool, help='the weights of support instances', default=False)

parser.add_argument('--task_lr', type=int, help='Task learning rate(里层)', default=7e-1)
parser.add_argument('--meta_lr', type=int, help='Meta learning rate(外层)', default=1)


args = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
#setup_seed(998244353)
#setup_seed(1000000007)
#setup_seed(7)
#setup_seed(1)
setup_seed(57)

noise_rate = args.noise_rate  # noise rate: [0,10]
B = args.B
N = args.N
K = args.K
Q = args.Q
Train_iter = args.Train_iter
Val_iter = args.Val_iter
Test_update_step = args.Test_update_step

print('----------------------------------------------------')
print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print(args)
print('----------------------------------------------------')

max_length = args.max_length
data_dir = {} # dataset
data_dir['noise_rate'] = noise_rate
data_dir['train'] = args.train
data_dir['val'] = args.val
data_dir['test'] = args.test

start_time = time.time()

miml=MyModel(args, B, N, K, max_length, data_dir)
train_model(miml, B, N, K, Q, data_dir,
    train_iter=Train_iter, val_iter=Val_iter, test_update_step=Test_update_step)

time_use=time.time()-start_time
h=int(time_use/3600)
time_use-=h*3600
m=int(time_use/60)
time_use-=m*60
s=int(time_use)
print('Totally used',h,'hours',m,'minutes',s,'seconds')
