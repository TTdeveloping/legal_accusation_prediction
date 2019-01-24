from model.Initialize import *
import torch.nn as nn
import torch.nn.functional as f
from DataUtils.common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class CNN(nn.Module):
    """
       CNN
    """

    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        Ci = 1
        kernel_nums = self.conv_filter_nums
        kernel_sizes = self.conv_filter_sizes
        paddingId = self.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)

        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)
        else:
            init_embedding(self.embed.weight)
        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)

        # cnn
        if self.wide_conv:
            print("Using Wide Convolution")
            self.conv = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), stride=(1, 1),
                                   padding=(K // 2, 0), bias=False) for K in kernel_sizes]
        else:
            print("Using Narrow Convolution")
            self.conv = [nn.Conv2d(in_channels=Ci, out_channels=kernel_nums, kernel_size=(K, D), bias=True) for K in kernel_sizes]

        for conv in self.conv:
            if self.use_cuda:
                conv.cuda()
        in_fea = len(kernel_sizes) * kernel_nums
        self.linear = nn.Linear(in_features=in_fea, out_features=C, bias=True)
        init_linear(self.linear)

    def forward(self, word, sentence_length):
        """
        :param word:
        :param sentence_length:
        :return:
        """
        # print(word)
        x = self.embed(word)
        # print(x)
        # exit()
        x = self.dropout_embed(x)
        x = x.unsqueeze(1)
        # print(x)
        # x = [f.relu(conv(x)).squeeze(3) for conv in self.conv]
        # print(x.size())
        conv_x = []
        # print("/????????")
        # print(self.conv)
        # exit()
        for conv in self.conv:
            conv_out = conv(x)
            # print(conv_out)
            # exit()
            conv_out_relu = f.relu(conv_out)
            # print(conv_out_relu)
            # exit()
            # print(conv_out.size())
            conv_out_relu = conv_out_relu.squeeze(3)
            # print(conv_out_relu.size())
            # exit()
            conv_x.append(conv_out_relu)
            # print(conv_x)

        # x = [f.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # x = [f.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_x]
        pool_x = []
        for i in conv_x:
            see_i = i.size(2)
            # print("following is result:")
            # print(see_i)
            max_pool_out = f.max_pool1d(i, see_i)
            # print("...........")
            # print(max_pool_out.size())
            max_pool_out = max_pool_out.squeeze(2)
            # print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
            # print(max_pool_out.size())
            pool_x.append(max_pool_out)
            # print(pool_x)
            # exit()

        x = torch.cat(pool_x, 1)
        # print(x.size())
        # exit()
        logit = self.linear(x)
        return logit








