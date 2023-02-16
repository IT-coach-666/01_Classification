import torch
from tqdm import tqdm
from utils import read_data, MyDataset
from config import parsers
from torch.utils.data import DataLoader
from model import MyModel
from torch.optim import AdamW
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time
import logging
import sys


from transformers import BertConfig, BertTokenizer


logging.basicConfig(level=logging.DEBUG,
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')


def run_test(args, tokenizer):
    device = args.device

    logging.info("Loading test data from: %s" % args.test_file)
    ls_test_text, ls_test_label = read_data(args.test_file)
    test_dataset = MyDataset(ls_test_text, ls_test_label, args.max_len, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    logging.info("Loading best model from: %s" % args.save_model_best)
    config = BertConfig.from_pretrained(args.save_model_best, num_labels=args.class_num)
    model = MyModel.from_pretrained(args.save_model_best, config=config)
    model.to(device)
    model.eval()
    logging.info("Begin to evaluate the model....")
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_text, batch_label in tqdm(test_dataloader):
            batch_label = batch_label.to(device)
            pred = model(batch_text[0].to(device), batch_text[1].to(device))
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


def run_predict(args, tokenizer, ls_text):
    start = time.time()
    logging.info("Loading best model from: %s" % args.save_model_best)
    config = BertConfig.from_pretrained(args.save_model_best, num_labels=args.class_num)
    model = MyModel.from_pretrained(args.save_model_best, config=config)
    model.to(args.device)
    model.eval()
    model_loaded_time = time.time()
    logging.info("模型加载耗时: %s" % (model_loaded_time - start))

    logging.info("模型预测结果：")

    for text in ls_text:
        token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
        mask = [1] * len(token_id) + [0] * (args.max_len + 2 - len(token_id))
        token_id = token_id + [0] * (args.max_len + 2 - len(token_id))
        token_id = torch.tensor(token_id).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        pred = model(token_id.to(args.device), mask.to(args.device))

        result = torch.argmax(pred, dim=1)
        result = result.cpu().numpy().tolist()
        classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
        classification_dict = dict(zip(range(len(classification)), classification))
        logging.info(f"文本：{text}\t预测的类别为：{classification_dict[result[0]]}")

    end = time.time()
    logging.info(f"数据预测耗时：{end - model_loaded_time} s")


def run_train(args, tokenizer):
    start = time.time()

    device = args.device 

    # jy: 获取训练集、验证集数据和标签列表;
    ls_train_text, ls_train_label = read_data(args.train_file)
    ls_dev_text, ls_dev_label = read_data(args.dev_file)

    # jy: 加载训练数据;
    train_dataset = MyDataset(ls_train_text, ls_train_label, args.max_len, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # jy: 加载验证集;
    dev_dataset = MyDataset(ls_dev_text, ls_dev_label, args.max_len, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # jy: 加载初始模型, 设定优化器、损失函数;
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=args.class_num)
    model = MyModel.from_pretrained(args.model_name_or_path, config=config)
    opt = AdamW(model.parameters(), lr=args.learn_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)


    acc_max = float("-inf")
    for epoch in range(args.epochs):
        loss_sum, count = 0, 0

        # jy: 训练模型;
        model.train()
        for batch_index, (batch_text, batch_label) in enumerate(train_dataloader):
            batch_label = batch_label.to(device)
            input_ids, attention_mask = batch_text[0].to(device), batch_text[1].to(device)
            pred = model(input_ids, attention_mask)

            loss = loss_fn(pred, batch_label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss
            count += 1

            # 打印内容
            if len(train_dataloader) - batch_index <= len(train_dataloader) % 1000 and count == len(train_dataloader) % 1000:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                logging.info(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0

            if batch_index % 1000 == 999:
                msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
                logging.info(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
                loss_sum, count = 0.0, 0

        # jy: 将模型在验证集上检验, 基于该检验得分挑选最好的模型;
        model.eval() 
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch_text, batch_label in tqdm(dev_dataloader):
                batch_label = batch_label.to(device)
                
                pred = model(batch_text[0].to(device), batch_text[1].to(device))

                pred = torch.argmax(pred, dim=1).cpu().numpy().tolist()
                label = batch_label.cpu().numpy().tolist()

                all_pred.extend(pred)
                all_true.extend(label)

        acc = accuracy_score(all_pred, all_true)
        logging.info(f"dev acc:{acc:.4f}")
        if acc > acc_max:
            logging.info("%s, %s" % (str(acc), str(acc_max)))
            acc_max = acc
            model.save_pretrained(args.save_model_best)
            logging.info("当前 epoch (%s) 模型取得了最佳效果, 已保存该模型" % epoch)

    # jy: 保存最后一轮训练的模型
    model.save_pretrained(args.save_model_last)

    end = time.time()
    logging.info(f"运行时间：{(end-start)/60%60:.4f} min")

    # jy: 在测试数据上进行测试;
    run_test(args, tokenizer)


if __name__ == "__main__":
    # jy: 解析配置参数;
    args = parsers()
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    if args.is_do_test_only:
        run_test(args, tokenizer)
        # jy: 提前退出程序;
        sys.exit(1)

    if args.is_do_predict_only:
        ls_predict_text = ["我们一起去打篮球吧！", "我喜欢踢足球！", "沈腾和马丽的新电影《独行月球》很好看",
                           "昨天玩游戏，完了一整天", "现在的高考都已经开始分科考试了。",
                           "中方：佩洛西如赴台将致严重后果", "现在的股票基金趋势很不好"]
        run_predict(args, tokenizer, ls_predict_text)
        sys.exit(1)
    run_train(args, tokenizer)


