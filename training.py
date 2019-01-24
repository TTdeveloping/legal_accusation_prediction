from DataUtils.Optim import *
import torch.nn as nn
from DataUtils.utilss import *
from DataUtils.cail_eval import Eval, F1_measure, getFscore_Avg
import time
import random
import sys
from DataUtils.common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Train(object):
    def __init__(self, **kwargs):
        print("The Training Is Starting")
        self.train_iter = kwargs["train_iter"]
        self.dev_iter = kwargs["dev_iter"]
        self.test_iter = kwargs["test_iter"]
        self.model = kwargs["model"]
        self.config = kwargs["config"]
        # self.early_max_patience = self.config.early_max_patience
        self.optimizer = Optimizer(name=self.config.learning_algorithm, model=self.model, lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay, grad_clip=self.config.clip_max_norm)
        self.loss_function = nn.CrossEntropyLoss(size_average=True)
        print(self.optimizer)
        print(self.loss_function)
        self.best_score = Best_Result()
        self.train_iter_len = len(self.train_iter)
        # define accu eval

        self.accu_train_eval_micro, self.accu_dev_eval_micro, self.accu_test_eval_micro = Eval(), Eval(), Eval()
        self.accu_train_eval_macro, self.accu_dev_eval_macro, self.accu_test_eval_macro = [], [], []
        for i in range(self.config.label_num):
            self.accu_train_eval_macro.append(Eval())
            self.accu_dev_eval_macro.append(Eval())
            self.accu_test_eval_macro.append(Eval())

    def _get_model_args(self, batch_features):
        """
        :param batch_features: batch instance
        :return:
        """
        word = batch_features.word_features
        # mask = word > 0
        sentence_length = batch_features.sentence_length
        labels = batch_features.label_features
        batch_size = batch_features.batch_length
        return word, sentence_length, labels, batch_size

    def train(self):
        epochs = self.config.epochs
        for epoch in range(1, epochs + 1):
            print("\n## The {} epoch,All {} epochs ! ##".format(epoch, epochs))
            start_time = time.time()
            random.shuffle(self.train_iter)
            self.model.train()

            steps = 1
            backword_count = 0
            self.optimizer.zero_grad()
            for batch_count, batch_features in enumerate(self.train_iter):  # train_iter里边是一个个batch特征()
                # print(batch_count)
                # print(batch_features)
                backword_count += 1
                # word, sentence_length, labels, batch_size = self._get_model_args(batch_features)
                # print(word)
                # print(labels)
                accu = self.model(batch_features.word_features, batch_features.sentence_length)
                accu_logit = accu.view(accu.size(0) * accu.size(1), accu.size(2))
                loss_accu = self.loss_function(accu_logit, batch_features.label_features)
                loss_accu.backward()  # 后向传播计算每一层的误差
                self.optimizer.step()  # 更新模型参数
                self.optimizer.zero_grad()  # 更新之后梯度清空，在更新之后的参数之上进行训练
                steps += 1
                if (steps - 1) % self.config.log_interval == 0:
                    self.accu_train_eval_micro.clear_PRF()
                    for i in range(self.config.label_num):  self.accu_train_eval_macro[i].clear_PRF()
                    F1_measure(accu, batch_features.label_features, self.accu_train_eval_micro,
                               self.accu_train_eval_macro, cuda=self.config.use_cuda)
                    (accu_p_avg, accu_r_avg, accu_f_avg), _, _ = getFscore_Avg(self.accu_train_eval_micro,
                                                                               self.accu_train_eval_macro, accu.size(1))
                    # print((accu_p_avg, accu_r_avg, accu_f_avg))
                    sys.stdout.write(
                        "\nbatch_count = [{}/{}] ,"
                        " loss_accu is {:.6f}, "
                        "[accu-ACC is {:.6f}%]".format(batch_count + 1,
                                                       self.train_iter_len, loss_accu.data[0], accu_p_avg))

            end_time = time.time()
            print("\nTrain Time {:.3f}".format(end_time - start_time), end="")
            self.eval(model=self.model, epoch=epoch, config=self.config)

    def eval(self, model, epoch, config):
        """
        :param model: nn model
        :param epoch:  epoch
        :param config:  config
        :return:
        """
        self.accu_dev_eval_micro.clear_PRF()
        for i in range(self.config.label_num): self.accu_dev_eval_macro[i].clear_PRF()
        eval_start_time = time.time()
        self._eval_batch(self.dev_iter, model, self.accu_dev_eval_micro, self.accu_dev_eval_macro,
                         self.best_score, epoch, config, test=False)
        eval_end_time = time.time()
        print("Dev Time {:.3f}".format(eval_end_time - eval_start_time))

        self.accu_test_eval_micro.clear_PRF()
        for i in range(self.config.label_num): self.accu_test_eval_macro[i].clear_PRF()
        eval_start_time = time.time()
        self._eval_batch(self.test_iter, model, self.accu_test_eval_micro, self.accu_test_eval_macro,
                         self.best_score, epoch, config, test=True)
        eval_end_time = time.time()
        print("Test Time {:.3f}".format(eval_end_time - eval_start_time))

    def _eval_batch(self, data_iter, model, accu_eval_micro, accu_eval_macro, best_score, epoch, config, test=False):
        """
        :param data_iter:
        :param model:
        :param accu_eval_micro:
        :param accu_eval_macro:
        :param best_score:
        :param epoch:
        :param config:
        :param test:
        :return:
        """
        model.eval()
        for batch_count, batch_features in enumerate(data_iter):
            accu = model(batch_features.word_features, batch_features.sentence_length)
            F1_measure(accu, batch_features.label_features, accu_eval_micro, accu_eval_macro, cuda=config.use_cuda)

        # get f-score
        # accu_p, accu_r, accu_f = getFscore_Avg(accu_eval_micro, accu_eval_macro, accu.size(1))
        accu_macro_micro_avg, accu_micro, accu_macro = getFscore_Avg(accu_eval_micro, accu_eval_macro, accu.size(1))
        accu_p, accu_r, accu_f = accu_macro_micro_avg
        accu_p_ma, accu_r_ma, accu_f_ma = accu_macro
        accu_p_mi, accu_r_mi, accu_f_mi = accu_micro

        p, r, f = accu_p, accu_r, accu_f
        # p, r, f = law_p, law_r, law_f

        test_flag = "Test"
        if test is False:
            print()
            test_flag = "Dev"
            best_score.current_dev_score = f
            if f >= best_score.best_dev_score:
                best_score.best_dev_score = f
                best_score.best_epoch = epoch
                best_score.best_test = True
        if test is True and best_score.best_test is True:
            best_score.p = p
            best_score.r = r
            best_score.f = f
        print("{}:".format(test_flag))
        print("Macro_Micro_Avg ===>>> ")
        print("Eval: accu    --- Precision = {:.6f}%  Recall = {:.6f}% , F-Score = {:.6f}%".format(accu_p, accu_r,
                                                                                                   accu_f))
        print("Macro ===>>> ")
        print("Eval: accu    --- Precision = {:.6f}%  Recall = {:.6f}% , F-Score = {:.6f}%".format(accu_p_ma, accu_r_ma,
                                                                                                   accu_f_ma))
        print("Micro ===>>> ")
        print("Eval: accu    --- Precision = {:.6f}%  Recall = {:.6f}% , F-Score = {:.6f}%".format(accu_p_mi, accu_r_mi,
                                                                                                   accu_f_mi))
        if test is True:
            print("The Current Best accu Dev F-score: {:.6f}, Locate on {} Epoch.".format(best_score.best_dev_score,
                                                                                          best_score.best_epoch))
        if test is True:
            best_score.best_test = False

    # def getAcc(self, logit, target, batch_size):  # 这个主要对比预测结果和金标值，看预测和金标一样的有多少，算出准确率
    #     """
    #     :param logit:  model predict(output)
    #     :param target:  actual value
    #     :param batch_size:
    #     :return:+
    #
    #     """
    #     corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    #     print("the following is my request；")
    #     print("logit", logit)
    #     print(torch.max(logit, 1))  # max(logit,1)是一行一行的比较找出最大值，如果写1的话，就是一列一列的比较取出最大值
    #     这个函数的输出结果是在找到每行的最大值的时候，把最大值输出并且输出最大值所在的列。比如0,1
    #     print("Result:")
    #     print(torch.max(logit, 1)[1].view(target.size()).data)
    #     print("Target:")
    #     print(target.data)
    #     print("aaa")
    #     print("bijiao")
    #     print((torch.max(logit, 1)[1].view(target.size()).data == target.data))  # 每一行用预测结果和金标值进行比较，
    #     相同的话在每一行记1
    #     print("sum()")
    #     print((torch.max(logit, 1)[1].view(target.size()).data == target.data).sum())
    #     exit()
        # accuracy = float(corrects) / batch_size * 100.0
        # return accuracy
#
#










