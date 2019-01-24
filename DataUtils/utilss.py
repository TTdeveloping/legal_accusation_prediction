from DataUtils.common import *
import random
import torch
torch.manual_seed(seed_num)
random.seed(seed_num)


class Best_Result:
    """
    Best_Result
    """
    def __init__(self):
        self.current_dev_score = -1
        self.best_dev_score = -1
        self.best_score = -1
        self.best_epoch = 1
        self.best_test = False
        self.early_current_patience = 0
        self.p = -1
        self.r = -1
        self.f = -1


def torch_max(output):
    """
    :param output: batch * seq_len * label_num
    :return:
    """
    # print(output)
    batch_size = output.size(0)
    # print(output)
    _, arg_max = torch.max(output, dim=2)
    # print(_)
    # exit()
    label = []
    for i in range(batch_size):
        label.extend(arg_max[i].cpu().data)
        # label.append(arg_max[i].cpu().data.numpy())
    return label
