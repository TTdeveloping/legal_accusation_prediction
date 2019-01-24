import random
import sys
from Dataload.Instance import Instance
import re
import torch
from DataUtils.common import *
import json
import jieba
torch.manual_seed(seed_num)
random.seed(seed_num)


class DataLoaderHelp(object):
    """
    DataLoaderHelp
    """

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _normalize_word(word):
        """
        :param word:
        :return:
        """
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0'
            else:
                new_word += char
        return new_word

    @staticmethod
    def _sort(insts):
        """
        :param insts:
        :return:
        """
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.words_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=True)
        for key, value in dict:
            sorted_insts.append(insts[key])
        print("Sort Finished.")
        return sorted_insts


class DataLoader(DataLoaderHelp):
    def __init__(self, path, shuffle, config):
        """
        :param path:
        :param shuffle:
        :param config:
        :return:
        """
        print("Loading data:......")
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle

    def dataload(self):
        """

        :return:
        """
        path = self.path
        shuffle = self.shuffle
        assert isinstance(path,  list,), "path must be in list"  # instance()指定对象是某种特定的形式
        print('data path{}'.format(path))
        for data_id in range(len(path)):
            print("load data form:{}".format(path[data_id]))
            insts = self._Load_Each_Data(path=path[data_id], shuffle=shuffle)  # 每一句话一个类，每一个类封装着单词列表，标签列表，词表长度
            if shuffle is True and data_id == 0:
                print("shuffle train data......")
                random.shuffle(insts)
            self.data_list.append(insts)
            # return train/dev/test data
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]

    def _Load_Each_Data(self, path=None, shuffle=False):
        """
        :param path:
        :param shuffle:
        :return:
        """
        assert path is not None, "the data is not allow empty"
        insts = []
        now_lines = 0

        with open(path, encoding="utf-8") as f:
            for line in f.readlines():
                now_lines += 1
                if now_lines % 200 == 0:
                    sys.stdout.write("\rNow is reading the {}line".format(now_lines))
                if line == "\n":
                    print("This is the Empty......")
                inst = Instance()
                line = json.loads(line)
                word = line["fact"][:500]
                word = word.split(" ")
                label = line["meta"]["accusation"]
                inst.fact = word
                inst.accusation_labels = label
                inst.fact_size = len(inst.fact)
                inst.accusation_labels_size = len(inst.accusation_labels)
                insts.append(inst)

                if len(insts) == self.max_count:
                    break
            sys.stdout.write("\rNow reading the {}lines\t".format(now_lines))
        return insts
