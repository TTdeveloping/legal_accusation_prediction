import torch.nn as nn
from model.Initialize import *
import torch.nn.functional as f


class GRU(nn.Module):
    def __init__(self, **kwargs):
        super(GRU, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        paddingId = self.paddingId
        self.embed = nn.Embedding(V, D, padding_idx=paddingId)  # 这是一种随机初始化embed的方式

        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)  # 训练好的文件的embed
        else:
            init_embedding(self.embed.weight)  # 另一种随机初始化方式来初始化embedding
        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(D, hidden_size=self.lstm_hiddens, num_layers=self.lstm_layers, bidirectional=True,
                          batch_first=True, bias=True)
        self.linear = nn.Linear(self.lstm_hiddens * 2, C)
        init_linear(self.linear)

    def forward(self, word, sentence_length):
        # print(word)  # 假如是16x30(只是个形式,因为每个batch最大句子长度不一样)
        x = self.embed(word)
        # print(x)  # 演变成16x30x300形式
        x = self.dropout_embed(x)
        # print(x)
        x, _ = self.gru(x)  # x是hidden, _是cell
        # print(x.size())
        # exit()
        x = x.permute(0, 2, 1)
        # print(x)
        x = self.dropout(x)
        x = f.max_pool1d(x, x.size(2)).squeeze(2)
        # print(x.size())
        x = f.tanh(x)
        logit = self.linear(x)
        # print(logit)
        return logit






