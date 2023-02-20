import argparse
import os.path


def parsers():
    #class_num = 10
    class_num = 2

    device_num = 3

    # jy: 基于不同的类别数, 选择不同的文件夹中的数据(需提前规划好);
    dir_data = "./data_%s_class" % class_num

    dir_out = "./model_out"

    #is_last_layer_only = True
    is_last_layer_only = False
    suffix_ = "lastLayer" if is_last_layer_only else "encodeLayer"

    parser = argparse.ArgumentParser(description="Bert model of argparse")
    # jy: 是否使用 Bert 的最后一层隐层向量;
    parser.add_argument("--last_layer_only", type=bool, default=is_last_layer_only,
                        help="选择模型")

    parser.add_argument("--train_file", type=str, default=os.path.join(dir_data, "train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join(dir_data, "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join(dir_data, "test.txt"))
    parser.add_argument("--classification", type=str, default=os.path.join(dir_data, "class.txt"))
    parser.add_argument("--model_name_or_path", type=str, 
                        #default="/data/pharm_paper_search-vecalign-LASER/jy_model/bert-base-chinese/",
                        default="/data/pharm_paper_search-vecalign-LASER/jy_model/bert-base-uncased/",
                        help="bert 预训练模型")

    # jy: 是否仅跑测试结果, 不进行训练;
    parser.add_argument("--is_do_test_only", type=bool, default=False)
    #parser.add_argument("--is_do_test_only", type=bool, default=True)
    # jy: 是否仅进行 predict, 不进行训练;
    parser.add_argument("--is_do_predict_only", type=bool, default=False)
    #parser.add_argument("--is_do_predict_only", type=bool, default=True)

    parser.add_argument("--class_num", type=int, default=class_num, help="分类数")
    parser.add_argument("--max_len", type=int, default=512, help="句子的最大长度")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)

    # jy: GPU 编号, 如果使用 cpu, 则对应的值设置为 "cpu"
    parser.add_argument("--device", type=str, default="cuda:%d" % device_num)

    parser.add_argument("--learn_rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2, help="失活率")
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help="TextCnn 的卷积核大小")
    parser.add_argument("--num_filters", type=int, default=2, help="TextCnn 的卷积输出")
    parser.add_argument("--encode_layer", type=int, default=12, help="bert 层数")
    parser.add_argument("--hidden_size", type=int, default=768, help="bert 层输出维度")
    parser.add_argument("--save_model_best", type=str, default=os.path.join(dir_out, "best_model_%s.pth" % suffix_))
    parser.add_argument("--save_model_last", type=str, default=os.path.join(dir_out, "last_model_%s.pth" % suffix_))

    args = parser.parse_args()
    return args
