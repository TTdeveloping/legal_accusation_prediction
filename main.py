import os
import datetime
import argparse
import Config.config as configurable
import torch
from DataUtils.mainshelp import *
from Dataload.dataload_SST_binary import *
from DataUtils.common import *
from training import Train
torch.manual_seed(seed_num)
random.seed(seed_num)


def start_train(train_iter, dev_iter, test_iter, model, config):
    """
    :param train_iter:  train batch data iterator
    :param dev_iter:  dev batch data iterator
    :param test_iter:  test batch data iterator
    :param model:  nn model
    :param config:  config
    :return:  None
    """
    t = Train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)
    t.train()


def main():
    config.mulu = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # config.add_args(key="mulu", value=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    config.save_dir = os.path.join(config.save_direction, config.mulu)
    # exit()
    if not os.path.isdir(config.save_dir): os.makedirs(config.save_dir)

    train_iter, dev_iter, test_iter, alphabet = load_data(config=config)

    # get params
    get_params(config=config, alphabet=alphabet)

    # save dictionary
    save_dictionary(config=config)

    model = load_model(config)
    print(model)

    if config.train is True:
        start_train(train_iter, dev_iter, test_iter, model, config)
        exit()


def parse_argument():
    """
    :argument
    :return:
    """
    parser = argparse.ArgumentParser(description="NER & POS")
    parser.add_argument("-c", "--config", dest="config_file", type=str, default="./Config/config.cfg",
                        help="config path")
    parser.add_argument("--train", dest="train", action="store_true", default=True, help="train model")
    parser.add_argument("-p", "--process", dest="process", action="store_true", default=True, help="data process")
    parser.add_argument("-t", "--test", dest="test", action="store_true", default=False, help="test model")
    parser.add_argument("--t_model", dest="t_model", type=str, default=None, help="model for test")
    parser.add_argument("--t_data", dest="t_data", type=str, default=None,
                        help="data[train dev test None] for test model")
    parser.add_argument("--predict", dest="predict", action="store_true", default=False, help="predict model")
    args = parser.parse_args()
    # print(vars(args))
    config = configurable.Configurable(config_file=args.config_file)
    config.train = args.train
    config.process = args.process
    config.test = args.test
    config.t_model = args.t_model
    config.t_data = args.t_data
    config.predict = args.predict
    # config
    if config.test is True:
        config.train = False
    if config.t_data not in [None, "train", "dev", "test"]:
        print("\nUsage")
        parser.print_help()
        print("t_data : {}, not in [None, 'train', 'dev', 'test']".format(config.t_data))
        exit()
    print("***************************************")
    print("Data Process : {}".format(config.process))
    print("Train model : {}".format(config.train))
    print("Test model : {}".format(config.test))
    print("t_model : {}".format(config.t_model))
    print("t_data : {}".format(config.t_data))
    print("predict : {}".format(config.predict))
    print("***************************************")

    return config


if __name__ == '__main__':
    print('Process ID{},Process Parent ID{}'.format(os.getpid(), os.getppid()))
    config = parse_argument()
    if config.use_cuda is True:
        print("use GPU training now:")
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print('starting init cuda seed', torch.cuda.initial_seed())

    main()


