import torch.nn as nn
from model.Initialize import *
import torch.nn.functional as f


class CNN_BiLSTM(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_BiLSTM, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        Ci = 1
        Cn = self.conv_filter_nums
        Cs = self.conv_filter_sizes
        self.embed = nn.Embedding(V, D, padding_idx=self.paddingId)
        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)
        else:
            init_embedding(self.embed.weight)

        # cnn
        self.conv1 = [nn.Conv2d(Ci, Cn, (K, D), stride=(1, 1), padding=(K // 2, 0), bias=False) for K in Cs]
        for conv in self.conv1:
            if self.use_cuda:
                conv.cuda()

        # bilstm
        self.bilstm = nn.LSTM(D, hidden_size=self.lstm_hiddens, num_layers=self.lstm_layers, bidirectional=True,
                              batch_first=True, bias=True)

        # linear
        L = len(Cs) * Cn + self.lstm_hiddens * 2
        self.linear1 = nn.Linear(L, L//2)
        self.linear2 = nn.Linear(L//2, C)

        # dropout
        self.dropout = nn.Dropout(self.dropout)
        self.dropout_emb = nn.Dropout(self.dropout_emb)

    def forward(self, word, sentence_length):
        embed = self.embed(word)
        # CNN
        x = embed.unsqueeze(1)
        conv_x = []
        for conv in self.conv1:
            conv_out = conv(x)
            conv_out_relu = f.relu(conv_out)
            conv_out_relu = conv_out_relu.squeeze(3)
            conv_x.append(conv_out_relu)
        pool_x = [f.max_pool1d(i, i.size(2)).squeeze(2)for i in conv_x]
        cnn_x = torch.cat(pool_x, 1)
        cnn_x = self.dropout(cnn_x)

        # lstm
        lstm_x = self.dropout(embed)
        lstm_x = lstm_x.permute(0, 2, 1)
        lstm_x = self.dropout_emb(lstm_x)
        lstm_x = f.max_pool1d(lstm_x, lstm_x.size(2)).squeeze(2)
        lstm_x = f.tanh(lstm_x)

        # cat
        cnn_lstm = torch.cat((cnn_x, lstm_x), 1)

        # linear
        cnn_lstm = self.linear1(cnn_lstm)
        cnn_lstm = self.linear2(cnn_lstm)
        # out
        logit = cnn_lstm
        return logit






























