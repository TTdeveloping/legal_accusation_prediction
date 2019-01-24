from model.Cnn import CNN
import torch.nn as nn
from model.bilstm import BiLSTM
import torch
from DataUtils.common import *
from model.gru import *
from model.cnn_bilstm import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Joint(nn.Module):
    """
    Sequence_Label
    """
    def __init__(self, config):
        super(Joint, self).__init__()
        self.config = config
        # embed
        self.embed_num = config.embed_num
        self.embed_dim = config.embed_dim
        self.label_num = config.label_num
        self.paddingId = config.paddingId
        # dropout
        self.dropout_emb = config.dropout_emb
        self.dropout = config.dropout
        # lstm
        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_layers = config.lstm_layers
        # pre train
        self.pretrained_embed = config.pretrained_embed
        self.pretrained_weight = config.pretrained_weight
        # cnn param
        self.wide_conv = config.wide_conv
        self.conv_filter_sizes = self._conv_filter(config.conv_filter_sizes)
        self.conv_filter_nums = config.conv_filter_nums
        self.use_cuda = config.use_cuda

        if self.config.model_bilstm:
            self.model = BiLSTM(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.label_num,
                                paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                                lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                                use_cuda=self.use_cuda)

    @staticmethod
    def _conv_filter(str_list):
        """
        :param str_list:
        :return:
        """
        int_list = []
        str_list = str_list.split(",")
        for str in str_list:
            int_list.append(int(str))
        return int_list

    def forward(self, word, sentence_length, train=False):
        """
        :param word:
        :param sentence_length:
        :param train:
        :return:
        """

        model_output = self.model(word, sentence_length)
        return model_output








