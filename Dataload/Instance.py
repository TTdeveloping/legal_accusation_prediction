import torch
from DataUtils.common import *
import random
torch.manual_seed(seed_num)
random.seed(seed_num)


class Instance():
    def __init__(self):
        self.fact = []
        self.accusation_labels = []

        self.fact_size = 0
        self.accusation_labels_size = 0

        self.fact_index = []
        self.accusation_labels_index = []

