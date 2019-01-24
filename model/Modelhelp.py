import torch
import random
from DataUtils.common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


def prepare_pack_padded_sequence(input_words, seq_lengths, use_cuda=False, descending=True):
    """
    :param input_words:
    :param seq_lengths:
    :param use_cuda:
    :param descending:
    :return:
    """
    sorted_seq_lengths, indices = torch.sort(torch.LongTensor(seq_lengths), descending=descending)
    # print("???***" * 40)
    # print(sorted_seq_lengths)
    # print("******" * 40)
    # print(indices)
    if use_cuda is True:
        sorted_seq_lengths, indices = sorted_seq_lengths.cuda(), indices.cuda()
    _, desorted_indices = torch.sort(indices, descending=False)
    # print("???***" * 40)
    # print(_)
    # print("******" * 40)
    # print(desorted_indices)
    sorted_inputs_words = input_words[indices]
    return sorted_inputs_words, sorted_seq_lengths.cpu().numpy(), desorted_indices
