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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 保证底层算法的确定性，提升可复现程度


class BERT(nn.Module):
    def __init__(self, args, blank_padding=True):
        super(BERT, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.n_way = args.N
        self.max_length = args.max_length
        self.blank_padding = blank_padding
        # 得到预训练好的BERT模型参数和映射表
        self.pretrained_path = 'bert-base-uncased'
        if os.path.exists(self.pretrained_path):
            config = BertConfig.from_pretrained(os.path.join(self.pretrained_path, 'bert-base-uncased-config.json'))
            self.bert = BertModel.from_pretrained(os.path.join(self.pretrained_path, 'bert-base-uncased-pytorch_model.bin'), config=config)
            self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.pretrained_path, 'bert-base-uncased-vocab.txt'))
        else:
            self.bert = BertModel.from_pretrained(self.pretrained_path)
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)

    def forward(self, inputs, is_classname=False):
        if is_classname is True:
            tokens, att_masks, outputs = [], [], []
            for _ in inputs:
                token, att_mask = self.tokenize_classname(_)
                tokens.append(token)
                att_masks.append(att_mask)
            token = torch.cat([t for t in tokens], 0)  # [N*K,max_length] 就是消了一个没用的维度
            att_mask = torch.cat([a for a in att_masks], 0)  # [N*K,max_length]
            sequence_output = self.bert(token, attention_mask=att_mask)  # [N*K,max_length,bert_size]
            # print("sequence_output.shape", sequence_output.shape)
            # print(sequence_output)
            # ebd, (hn, cn) = self.lstm(sequence_output)  # [N*K, max_length, 256]
            # outputs = self.seq(ebd.transpose(1, 2).contiguous()).squeeze(-1)  # [N*K, 256]
            return sequence_output[:, 0, :]

        else:
            tokens, att_masks, head_poses, outputs = [], [], [], []
            for _ in inputs:
                token, att_mask, head_pos = self.tokenize(_)
                tokens.append(token)
                att_masks.append(att_mask)
                head_poses.append(head_pos)
            token = torch.cat([t for t in tokens], 0)  # [N*K,max_length] 就是消了一个没用的维度
            att_mask = torch.cat([a for a in att_masks], 0)  # [N*K,max_length]
            # sequence_output,pooled_output=self.bert(token,attention_mask=att_mask)
            sequence_output = self.bert(token, attention_mask=att_mask)  # [N*K,max_length,bert_size]
            # ebd, (hn, cn) = self.lstm(sequence_output)  # [N*K, max_length, 256]
            # outputs = self.seq(ebd.transpose(1, 2).contiguous()).squeeze(-1)  # [N*K, 256]
            for i in range(token.size(0)):
                outputs.append(self.entity_start_state(head_poses[i], sequence_output[i]))
            outputs = torch.cat([o for o in outputs], 0)
            # outputs = self.fc(outputs)  # [N*K,bert_size*2]
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
            re_tokens += self.tokenizer.tokenize(token)  # 把词分成pieces
            if cur_pos == pos_head[-1] - 1:
                re_tokens.append('[unused2]')
            if cur_pos == pos_tail[-1] - 1:
                re_tokens.append('[unused3]')
            cur_pos += 1
        re_tokens.append('[SEP]')

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        if self.blank_padding:  # 将长度padding成max_length
            while len(indexed_tokens) < self.max_length: indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)
        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1  # 就是标记哪些id是padding上去的（标记句子的实际长度）
        if self.cuda:
            indexed_tokens, att_mask = indexed_tokens.cuda(), att_mask.cuda()

        # head pos是一个列表，列表里两个元素，分别是两个位置（int），标记两个实体的起始标志位
        return indexed_tokens, att_mask, head_pos  # both [1,max_length]

    def tokenize_classname(self, inputs):
        tokens = inputs

        re_tokens, cur_pos = ['[CLS]', ], 0
        for token in tokens:
            token = token[0].lower()
            re_tokens += self.tokenizer.tokenize(token)  # 把词分成pieces
            cur_pos += 1
        re_tokens.append('[SEP]')

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        if self.blank_padding:  # 将长度padding成max_length
            while len(indexed_tokens) < self.max_length: indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)
        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :avai_len] = 1  # 就是标记哪些id是padding上去的（标记句子的实际长度）
        if self.cuda:
            indexed_tokens, att_mask = indexed_tokens.cuda(), att_mask.cuda()

        return indexed_tokens, att_mask  # both [1,max_length]


class MyModel(nn.Module):

    def __init__(self, args):
        super(MyModel, self).__init__()

        self.Batch = args.B
        self.n_way = args.N
        self.k_shot = args.K
        self.max_length = args.max_length
        self.coder = BERT(args)  # <[N*K,length], >[N*K, (BiLSTM)hidden_size * 2]
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True,
                            dropout=0.2)
        self.mlp = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768*2, 768),
            nn.ReLU()
        )
        self.cost = nn.CrossEntropyLoss()

    def forward(self, inputs, is_classname=False):  # inputs:[N*K, 768*2] or [N, 768]

        # ebd, (hn, cn) = self.lstm(inputs)  # -> [N*K, max_length, 512]
        # outputs = self.seq(ebd.transpose(1, 2).contiguous()).squeeze(-1)  # -> [N*K, 512]
        if is_classname is False:
            inputs = self.fc(inputs)

        outputs = self.mlp(inputs)

        return outputs

    def cloned_fc_dict(self):
        return {key: val.clone() for key, val in self.fc.state_dict().items()}

    def cloned_mlp_dict(self):
        return {key: val.clone() for key, val in self.mlp.state_dict().items()}

    def loss(self, logits, label):
        return self.cost(logits, label)

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class MyModel_Clone(nn.Module):

    def __init__(self, args):
        super(MyModel_Clone, self).__init__()

        self.Batch = args.B
        self.n_way = args.N
        self.k_shot = args.K
        self.max_length = args.max_length
        self.coder = BERT(args)  # <[N*K,length], >[N*K, (BiLSTM)hidden_size * 2]
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True,
                            dropout=0.2)
        self.mlp = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768*2, 768),
            nn.ReLU()
        )
        self.cost = nn.CrossEntropyLoss()

    def forward(self, inputs, is_classname=False):  # inputs:[N*K, 768*2] or [N, 768]

        # ebd, (hn, cn) = self.lstm(inputs)  # -> [N*K, max_length, 512]
        # outputs = self.seq(ebd.transpose(1, 2).contiguous()).squeeze(-1)  # -> [N*K, 512]
        if is_classname is False:
            inputs = self.fc(inputs)

        outputs = self.mlp(inputs)

        return outputs

    def cloned_fc_dict(self):
        return {key: val.clone() for key, val in self.fc.state_dict().items()}

    def cloned_mlp_dict(self):
        return {key: val.clone() for key, val in self.mlp.state_dict().items()}

    def loss(self, logits, label):
        return self.cost(logits, label)

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


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
            class_name = self.json_class_name[name][0] + " it means " + self.json_class_name[name][1]
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


def neg_dist(instances, class_proto):  # ins:[N*K, 256], cla:[N, 256]
    return -torch.pow(torch.pow(class_proto.unsqueeze(0) - instances.unsqueeze(1), 2).sum(-1), 0.5)


def get_dataloader(file_name, file_class, N, K, L, noise_rate):
    data_loader = data.DataLoader(
        dataset=FewRel(file_name, file_class, N, K, L, noise_rate),
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    return iter(data_loader)


def train_one_batch(batch, class_name0, support0, support_label, query0, query_label, mymodel, Test_update_step, task_lr, it,
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


def deep_copy(model1, model2):
    name_list = []
    for name in model1.state_dict():
        name_list.append(name)

    for name in model2.state_dict():
        if name in name_list:
            model2.state_dict()[name] = model1.state_dict()[name]

    return


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


def train_model(mymodel, mymodel_clone, args, val_step=100):

    n_way_k_shot = str(args.N) + '-way-' + str(args.K) + '-shot'
    print('Start training' + n_way_k_shot)

    cuda = torch.cuda.is_available()
    if cuda:
        mymodel = mymodel.cuda()
        mymodel_clone = mymodel_clone.cuda()

    data_loader = {}
    data_loader['train'] = get_dataloader(args.train, args.class_name_file, args.N, args.K, args.L, args.noise_rate)
    data_loader['val'] = get_dataloader(args.val, args.class_name_file, args.N, args.K, args.L, args.noise_rate)
    data_loader['test'] = get_dataloader(args.test, args.class_name_file, args.N, args.K, args.L, args.noise_rate)

    optim_params=[{'params': mymodel.coder.parameters(), 'lr': 5e-5}]
    optim_params.append({'params': mymodel.fc.parameters(), 'lr': args.task_lr})
    optim_params.append({'params': mymodel.mlp.parameters(), 'lr': args.task_lr})
    meta_optimizer = AdamW(optim_params, lr=1)

    # mymodel1_meta_opt = AdamW(mymodel.parameters(), lr=args.meta_lr)
    # mymodel2_task_opt = AdamW(mymodel.parameters(), lr=args.task_lr)

    best_acc, best_step, best_test_acc, best_test_step, best_val_loss, best_changed = 0.0, 0, 0.0, 0, 100.0, False
    iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0


    for it in range(args.Train_iter):
        meta_loss, meta_right = 0.0, 0.0
        # torch.save(mymodel2.state_dict(), 'model_checkpoint/checkpoint.{}th.tar'.format(it))
        for batch in range(args.B):
            mymodel.train()
            class_name, support, support_label, query, query_label = next(data_loader['train'])
            # [N, length], tokens:{[N*K,length]}, [1,N*K], tokens:{[N*L,length]}, [1,N*L]
            if cuda:
                support_label, query_label = support_label.cuda(), query_label.cuda()

            loss_s, right_s, query1, class_name1 = train_one_batch(batch, class_name, support, support_label, query, query_label, mymodel,
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

            # -----在Query上计算loss和acc-------
            loss_q, right_q = train_q(batch, class_name1, query1, query_label, mymodel_clone)
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

        if (it + 1) % val_step == 0:
            acc, val_loss = test_model(cuda, data_loader['val'], mymodel, args.Val_iter, args.Test_update_step, args.task_lr)
            print('[EVAL] | loss: {0:2.6f}, accuracy: {1:2.2f}%'.format(val_loss, acc * 100))
            if acc >= best_acc or val_loss <= best_val_loss:
                print('Best checkpoint!')
                torch.save(mymodel.state_dict(), 'model_checkpoint/checkpoint.{}th_best_model' + n_way_k_shot + '.tar'.format(it))
                best_acc, best_step, best_val_loss, best_changed = acc, (it + 1), val_loss, True

        torch.cuda.empty_cache()

    print("\n####################\n")
    print('Finish training model! Best acc: ' + str(best_acc) + ' at step ' + str(best_step))


def test_model(cuda, data_loader, mymodel, val_iter, test_update_step, task_lr, zero_shot=False):

    meta_loss = 0.0
    accs=0.0
    mymodel.eval()
    for it in range(val_iter):
        class_name, support, support_label, query, query_label = next(data_loader)
        if cuda:
            support_label, query_label = support_label.cuda(), query_label.cuda()

        '''First Step'''
        loss_s, right_s, query1, class_name1 = train_one_batch(0, class_name, support, support_label, query, query_label, mymodel, args.Test_update_step, args.task_lr, it)

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
        for name in mymodel.state_dict():
            name_list.append(name)

        for name in orderd_params:
            if name in name_list:
                mymodel.state_dict()[name].copy_(orderd_params[name])

        '''second-5th step'''
        for _ in range(5-1):
            loss_s, right_s, query1, class_name1 = train_one_batch(0, class_name, support, support_label, query,
                                                                   query_label, mymodel, args.Test_update_step,
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
            for name in mymodel.state_dict():
                name_list.append(name)

            for name in orderd_params:
                if name in name_list:
                    mymodel.state_dict()[name].copy_(orderd_params[name])

        # -----在Query上计算loss和acc-------
        loss_q, right_q = train_q(0, class_name1, query1, query_label, mymodel)
        meta_loss = meta_loss + loss_q
        accs += right_q

        if (it+1) % 100 == 0:

            print('step: {0:4} | val_loss:{1:3.6f}, val_accuracy: {2:3.2f}%'.format(it+1, meta_loss/(it+1), 100*accs/(it+1)))

        torch.cuda.empty_cache()

    return accs/val_iter, meta_loss/val_iter


def main(args):

    print('----------------------------------------------------')
    print("{}-way-{}-shot Few-Shot Relation Classification".format(args.N, args.K))
    print("Model: {}".format(args.Model))
    print("config:", args)
    print('----------------------------------------------------')
    start_time = time.time()

    mymodel = MyModel(args)
    mymodel_clone = MyModel_Clone(args)
    # for param in mymodel1.coder.parameters():
    #     param.requires_grad = False
    train_model(mymodel, mymodel_clone, args)



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
    parser.add_argument('--L', type=int, help='number of query per class', default=30)
    parser.add_argument('--noise_rate', type=int, help='noise rate, value range 0 to 10', default=0)
    parser.add_argument('--task_lr', type=int, help='Task learning rate(里层)', default=1)
    parser.add_argument('--meta_lr', type=int, help='Meta learning rate(外层)', default=1e-4)

    args = parser.parse_args()

    main(args)