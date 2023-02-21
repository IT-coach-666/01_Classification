# -*- coding:utf-8 -*-
# @author: 木子川
# @Data:   2022/8/1
# @Email:  m21z50c71@163.com

import argparse
import os.path


def parsers():
    #class_num = 2
    class_num = 10

    device_num = 2

    # jy: 基于不同的类别数, 选择不同的文件夹中的数据(需提前规划好);
    dir_data = "./data_%s_class" % class_num

    dir_out = "./model_out"

    parser = argparse.ArgumentParser(description="TextCNN model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join(dir_data, "train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join(dir_data, "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join(dir_data, "test.txt"))
    parser.add_argument("--classification", type=str, default=os.path.join(dir_data, "class.txt"))
    parser.add_argument("--data_pkl", type=str, default=os.path.join(dir_data, "dataset.pkl"))
    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=38)
    parser.add_argument("--embedding_num", type=int, default=50)

    # jy: 是否仅跑测试结果, 不进行训练;
    parser.add_argument("--is_do_test_only", type=bool, default=False)
    #parser.add_argument("--is_do_test_only", type=bool, default=True)
    # jy: 是否仅进行 predict, 不进行训练;
    parser.add_argument("--is_do_predict_only", type=bool, default=False)
    #parser.add_argument("--is_do_predict_only", type=bool, default=True)

    # jy: GPU 编号, 如果使用 cpu, 则对应的值设置为 "cpu"
    parser.add_argument("--device", type=str, default="cuda:%d" % device_num)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learn_rate", type=float, default=1e-3)
    parser.add_argument("--num_filters", type=int, default=2, help="卷积产生的通道数")
    parser.add_argument("--save_model_best", type=str, default=os.path.join(dir_out, "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join(dir_out, "last_model.pth"))
    args = parser.parse_args()
    return args

