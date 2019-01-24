import torch
from collections import OrderedDict
import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from DataUtils.common import *
import random
torch.manual_seed(seed_num)
random.seed(seed_num)


class Embed(object):
    """
    Embed
    """
    def __init__(self, path, words_dict, embed_type, pad):  # word_dict是word_alphabet.id2words
        self.embed_type_enum = ["zeros", "avg", "uniform", "nn"]
        self.path = path
        self.words_dict = words_dict
        self.embed_type = embed_type
        self.pad = pad
        if not isinstance(self.words_dict, dict):
            self.words_dict, self.words_list = self._list2dict(self.words_dict)
        if pad is not None:
            # print(self.words_dict)
            self.padID = self.words_dict[pad]
        self.dim, self.words_count = self._get_dim(path=self.path), len(self.words_dict)
        self.exact_count, self.fuzzy_count, self.oov_count = 0, 0, 0

    def _list2dict(self, convert_list):
        """
        :param convert_list: words_dict
        :return:
        """
        list_dict = OrderedDict()
        list_lower = []
        for index, word in enumerate(convert_list):
            list_lower.append(word.lower())
            # print(list_lower)
            # exit()
            list_dict[word] = index
        assert len(list_lower) == len(list_dict)
        return list_dict, list_lower

    def _get_dim(self, path):
        """
        :param path: glove.sentiment.conj.pretrained.txt
        :return:
        """
        embedding_dim = -1
        with open(path, encoding='utf-8') as f:
            for line in f:
                line_split = line.strip().split(' ')
                if len(line_split) == 1:
                    embedding_dim = line_split[0]
                    break
                elif len(line_split) == 2:
                    embedding_dim = line_split[1]
                    break
                else:
                    embedding_dim = len(line_split) - 1
                    break
        return embedding_dim

    def get_embed(self):
        """
        :param self:
        :return:
        """
        embed_dict = None
        if self.embed_type in self.embed_type_enum:
            embed_dict = self._read_file(path=self.path)  # embed_dict {‘单词’:'100维的数'}
        else:
            print("embed type is illegal,must be in {}".format(self.embed_type_enum))
            exit()
        embed = None
        if self.embed_type == "zeros":
            embed = self._zero_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        elif self.embed_type == "nn":
            embed = self._nn_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        elif self.embed_type == "uniform":
            embed = self._uniform_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        elif self.embed_type == "avg":
            embed = self._avg_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        self.info()
        return embed

        # self.info()
    def _zero_embed(self, embed_dict, words_dict):
        """
        :param embed_dict:
        :param words_dict:
        :return:
        """
        print("loading pre_train embedding by zeros for out of vocabulary.")
        embeddings = np.zeros((int(self.words_count), int(self.dim)))  # 初始化和embedding同样大小的矩阵
        # print(embeddings)
        # print(embeddings.size)
        # exit()
        # print("??????????????????"*20)
        # print(words_dict)
        # print("??????????????????"*20)
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    def _nn_embed(self, embed_dict, words_dict):
        """
        :param embed_dict:
        :param words_dict:
        :return:
        """
        print("loading pre_train embedding by nn.Embedding for out of vocabulary")
        embed = nn.Embedding(int(self.words_count), int(self.dim))
        init.xavier_uniform(embed.weight.data)
        embeddings = np.array(embed.weight.data)
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        final_embed = torch.from_numpy(embeddings).float()
        # print("please output the content:>>>>>>>>>>>>>>>>>>>")
        # print(final_embed)
        # exit()
        return final_embed

    def _uniform_embed(self, embed_dict, words_dict):
        """
        :param embed_dict:
        :param words_dict:
        """
        print("loading pre_train embedding by uniform for out of vocabulary.")
        embeddings = np.zeros((int(self.words_count), int(self.dim)))
        inword_list = {}
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        uniform_col = np.random.uniform(-0.25, 0.25, int(self.dim)).round(6)  # uniform
        for i in range(len(words_dict)):
            if i not in inword_list and i != self.padID:
                embeddings[i] = uniform_col
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    def _avg_embed(self, embed_dict, words_dict):
        """
        :param embed_dict:
        :param words_dict:
        """
        print("loading pre_train embedding by avg for out of vocabulary.")
        embeddings = np.zeros((int(self.words_count), int(self.dim)))
        inword_list = {}
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        sum_col = np.sum(embeddings, axis=0) / len(inword_list)  # avg
        for i in range(len(words_dict)):
            if i not in inword_list and i != self.padID:
                embeddings[i] = sum_col
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    def info(self):
        """
        :return:
        """
        total_count = self.exact_count +self.fuzzy_count
        print("words count {}, embed dim {}.".format(self.words_count,self.dim))
        print("exact rate {}/ {}".format(self.exact_count, self.words_count))
        print("fuzzy rate {}/ {}".format(self.fuzzy_count, self.words_count))
        print("INV rate {}/ {}".format(total_count, self.words_count))
        print("OOV rate {} / {}".format(self.oov_count,self.words_count))
        print("  OOV radio ===> {}%".format(np.round((self.oov_count / total_count) * 100, 2)))
        print(40 * "*")

    def _read_file(self, path):
        """
        :param path: pretrained.txt
        :return:
        """
        embed_dict = {}
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
            # print(lines)
            # exit()
            lines = tqdm.tqdm(lines)  # 进度条
            for line in lines:
                values = line.strip().split(' ')
                # print(values)
                if len(values) == 1 or len(values) == 2 or len(values) == 3:
                    continue
                w = values[0]
                v = values[1:]
                # print(w)
                # print(v)
                embed_dict[w] = v
                # print(embed_dict)
        return embed_dict










