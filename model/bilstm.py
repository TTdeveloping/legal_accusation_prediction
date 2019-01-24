import torch.nn as nn
import torch
import random
from DataUtils.common import *
from model.Initialize import *
import torch.nn.functional as f
from model.Modelhelp import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .Initialize import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class BiLSTM(nn.Module):

    def __init__(self, **kwargs):
        super(BiLSTM, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        paddingId = self.paddingId
        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        self.mlp_out_size = 2

        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)
        # else:
        #     print(self.embed.weight)
        #     exit()
        #     init_embedding(self.embed.weight)

        # accu
        # print(self.label_num)
        self.accu_embed = nn.Embedding(self.label_num, 200)
        self.accu_weight = self.accu_embed.weight

        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)
        self.bilstm = nn.LSTM(input_size=D, hidden_size=self.lstm_hiddens, num_layers=self.lstm_layers,
                              bidirectional=True, batch_first=True, bias=True)
        self.nonLinear = nn.Linear(in_features=self.lstm_hiddens * 2, out_features=self.mlp_out_size, bias=True)
        # init_linear(self.linear)

        self.accu_law_nonlinear = nn.Linear(in_features=200, out_features=50, bias=True)
        init_linear_weight_bias(self.accu_law_nonlinear)

        # accu and law
        self.linear = nn.Linear(in_features=50, out_features=2, bias=True)
        init_linear_weight_bias(self.linear)

    def accu_forward(self, x):
        """
        :param x: accu_x 16x2的矩阵
        :return:
        """
        # print(x.size())
        x = x.unsqueeze(1)  # 16x2变成16x1x2
        print(x)
        x = x.repeat(1, self.label_num, 1)  # 16x167x2
        print(x)
        exit()
        x = torch.mul(x, self.accu_weight)
        x = f.tanh(self.accu_law_nonlinear(x))
        x = f.tanh(x)
        accu = self.linear(x)
        return accu

    def forward(self, word, sentence_length):
        """
        :param word:
        :param sentence_length:
        :return:
        """
        word, sentence_length, desorted_indices = prepare_pack_padded_sequence(word, sentence_length,
                                                                               use_cuda=self.use_cuda)
        x = self.embed(word)
        x = self.dropout_embed(x)
        # print(x)
        packed_embed = pack_padded_sequence(x, sentence_length, batch_first=True)
        x, _ = self.bilstm(packed_embed)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[desorted_indices]  # 16x197x300
        x = x.permute(0, 2, 1)   # 16x300x197
        # x = self.dropout(x)
        x = f.max_pool1d(x, x.size(2)).squeeze(2)  # 16x300 maxpooling每句话只留最具有代表性的一个字
        # x = f.tanh(x)
        accu_x = self.nonLinear(x)  # 输入300维变成2维  16x2
        accu_x = f.tanh(self.nonLinear(x))  # 16x2
        accu = self.accu_forward(accu_x)
        return accu









