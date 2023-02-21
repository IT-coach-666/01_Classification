import os
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import read_data, built_curpus, TextDataset
from model import TextCNNModel
from config import parsers
import pickle as pkl
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import logging
import time


logging.basicConfig(level=logging.DEBUG,
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')



def run_test(args):
    device = args.device if torch.cuda.is_available() else "cpu"

    # jy: 模型训练时, 必然会存储 args.data_pkl 文件, 该文件记录词表以及其对应的 embedding;
    dataset = pkl.load(open(args.data_pkl, "rb"))
    word_2_index, words_embedding = dataset[0], dataset[1]

    ls_test_text, ls_test_label = read_data(args.test_file)
    test_dataset = TextDataset(ls_test_text, word_2_index, args.max_len, ls_test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TextCNNModel(words_embedding, args.max_len, args.class_num, args.num_filters).to(device)
    model.load_state_dict(torch.load(args.save_model_best))
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_text, batch_label in tqdm(test_dataloader):
            batch_text, batch_label = batch_text.to(device), batch_label.to(device)
            pred = model(batch_text)
            pred = torch.argmax(pred, dim=1)

            pred = pred.cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()

            all_pred.extend(pred)
            all_true.extend(label)

    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, average="micro")
    recall = recall_score(all_true, all_pred, average="micro")
    f1 = f1_score(all_true, all_pred, average="micro")

    logging.info(f"test dataset accuracy:{accuracy:.4f}\tprecision:{precision:.4f}\trecall:{recall:.4f}\tf1:{f1:.4f}")



def run_predict(args, ls_predict_text):
    device = args.device if torch.cuda.is_available() else "cpu"

    # jy: 模型训练时, 必然会存储 args.data_pkl 文件, 该文件记录词表以及其对应的 embedding;
    dataset = pkl.load(open(args.data_pkl, "rb"))
    word_2_index, words_embedding = dataset[0], dataset[1]

    # 加载模型
    model_path = args.save_model_best
    model = TextCNNModel(words_embedding, args.max_len, args.class_num, args.num_filters).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    logging.info("模型预测结果：")
    text_dataset = TextDataset(ls_predict_text, word_2_index, args.max_len)
    test_dataloader = DataLoader(text_dataset, batch_size=1, shuffle=False)
    for idx, batch_text in enumerate(test_dataloader):
        batch_text = batch_text.to(device)
        pred = model(batch_text)

        result = torch.argmax(pred, dim=1)
        result = result.cpu().numpy().tolist()
        classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
        classification_dict = dict(zip(range(len(classification)), classification))
        logging.info(f"文本：{ls_predict_text[idx]}\t预测的类别为：{classification_dict[result[0]]}")


ls_predict_text = [
"我们一起去打篮球吧！", 
"沈腾和马丽的新电影《独行月球》很好看", 
"昨天玩游戏，完了一整天",
"现在的高考都已经开始分科考试了。", 
"中方：佩洛西如赴台将致严重后果", 
"现在的股票基金趋势很不好"

]


if __name__ == "__main__":
    args = parsers()
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.is_do_test_only:
        run_test(args)
        sys.exit(1)

    if args.is_do_predict_only:
        run_predict(args, ls_predict_text)
        sys.exit(1)


    ls_train_text, ls_train_label = read_data(args.train_file)
    ls_dev_text, ls_dev_label = read_data(args.dev_file)

    if os.path.exists(args.data_pkl):
        dataset = pkl.load(open(args.data_pkl, "rb"))
        word_2_index, words_embedding = dataset[0], dataset[1]
    else:
        # jy: 注意, 该函数目前只适合于中文文本, 因为当前的处理逻辑是将每个字符当做一个 token;
        #     更加正规的方式是使用已有的 tokenize 工具进行 tokenize 处理后再基于相应的结果构
        #     造词表和初始化词向量;
        word_2_index, words_embedding = built_curpus(args, ls_train_text, args.embedding_num)

    train_dataset = TextDataset(ls_train_text, word_2_index, args.max_len, ls_train_label)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)

    dev_dataset = TextDataset(ls_dev_text, word_2_index, args.max_len, ls_dev_label)
    dev_loader = DataLoader(dev_dataset, args.batch_size, shuffle=False)

    model = TextCNNModel(words_embedding, args.max_len, args.class_num, args.num_filters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learn_rate)
    loss_fn = nn.CrossEntropyLoss()

    acc_max = float("-inf")
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0, 0
        for batch_index, (batch_text, batch_label) in enumerate(train_loader):
            batch_text, batch_label = batch_text.to(device), batch_label.to(device)
            pred = model(batch_text)

            loss = loss_fn(pred, batch_label)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss
            count += 1

            # 打印内容
            if len(train_loader) - batch_index <= len(train_loader) % 1000 and count == len(train_loader) % 1000:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                logging.info(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0

            if batch_index % 1000 == 999:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                logging.info(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0

        # jy: 每轮 epoch 进行一次开发集上的评估, 并保存当前最好模型;
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch_text, batch_label in tqdm(dev_loader):
                batch_text = batch_text.to(device)
                batch_label = batch_label.to(device)
                pred = model(batch_text)

                pred = torch.argmax(pred, dim=1)
                pred = pred.cpu().numpy().tolist()
                label = batch_label.cpu().numpy().tolist()

                all_pred.extend(pred)
                all_true.extend(label)

        acc = accuracy_score(all_pred, all_true)
        logging.info(f"dev acc:{acc:.4f}")

        if acc > acc_max:
            acc_max = acc
            torch.save(model.state_dict(), args.save_model_best)
            logging.info(f"以保存最佳模型")

    torch.save(model.state_dict(), args.save_model_last)

    run_test(args)
