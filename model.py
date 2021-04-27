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
from utils import neg_dist


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
        self.L = args.L
        self.lam = args.lam
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

    def loss(self, logits, label, support, class_name, NPM, isQ=False):
        loss_ce = self.cost(logits, label)

        if NPM is True:
            loss_npm = torch.tensor(0.0, requires_grad=True)
            if isQ is True:
                support_N = support.view((self.n_way, self.L, 256))
            else:
                support_N = support.view((self.n_way, self.k_shot, 256))
            for i, s in enumerate(support_N):
                dist = -neg_dist(s, class_name)  # [K, N]
                for j, d in enumerate(dist):
                    for k, di in enumerate(d):
                        loss_npm = loss_npm + torch.exp(d[i] - di)
            return loss_ce + self.lam * torch.log(loss_npm)
        else:
            return loss_ce

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
        self.L = args.L
        self.lam = args.lam
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

    def loss(self, logits, label, support, class_name, NPM, isQ=False):
        loss_ce = self.cost(logits, label)

        if NPM is True:
            loss_npm = torch.tensor(0.0, requires_grad=True)
            if isQ is True:
                support_N = support.view((self.n_way, self.L, 256))
            else:
                support_N = support.view((self.n_way, self.k_shot, 256))
            for i, s in enumerate(support_N):
                dist = -neg_dist(s, class_name)  # [K, N]
                for j, d in enumerate(dist):
                    for k, di in enumerate(d):
                        loss_npm = loss_npm + torch.exp(d[i] - di)
            return loss_ce + self.lam * torch.log(loss_npm)
        else:
            return loss_ce

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))