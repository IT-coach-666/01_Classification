# -*- coding:utf-8 -*-
# @author: 木子川
# @Data:   2022/8/8
# @Email:  m21z50c71@163.com

import argparse
import os.path


def parsers():
    #dir_data = "./data-all"
    dir_data = "./data"
    dir_out = "./model_out"
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join(dir_data, "train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join(dir_data, "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join(dir_data, "test.txt"))
    parser.add_argument("--classification", type=str, default=os.path.join(dir_data, "class.txt"))
    # jy: 模型名称或本地模型路径;
    parser.add_argument("--model_name_or_path", type=str,
                        #default="bert-base-chinese")
                        default="/data/pharm_paper_search-vecalign-LASER/jy_model/bert-base-chinese/")

    # jy: 是否仅跑测试结果, 不进行训练;
    parser.add_argument("--is_do_test_only", type=bool, default=False)
    #parser.add_argument("--is_do_test_only", type=bool, default=True)
    # jy: 是否仅进行 predict, 不进行训练;
    parser.add_argument("--is_do_predict_only", type=bool, default=False)
    #parser.add_argument("--is_do_predict_only", type=bool, default=True)

    # jy: 分类的类别数;
    parser.add_argument("--class_num", type=int, default=10)
    # jy: 序列的最大长度, 实际训练过程中的 batch 中的序列的长度会比该值大 2;
    parser.add_argument("--max_len", type=int, default=38)
    # jy: 训练时的 batch_size 大小;
    parser.add_argument("--batch_size", type=int, default=32)
    # jy: 训练轮数;
    parser.add_argument("--epochs", type=int, default=10)
    # jy: 学习率;
    parser.add_argument("--learn_rate", type=float, default=1e-5)
    # jy: GPU 编号, 如果使用 cpu, 则对应的值设置为 "cpu"
    parser.add_argument("--device", type=str, default="cuda:6")
    # jy: 最优模型的保存路径;
    parser.add_argument("--save_model_best", type=str, default=os.path.join(dir_out, "best_model"))
    # jy: 最后一轮训练完成后的模型保存路径;
    parser.add_argument("--save_model_last", type=str, default=os.path.join(dir_out, "last_model"))
    args = parser.parse_args()
    return args

