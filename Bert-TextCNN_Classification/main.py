import time
from tqdm import tqdm
from config import parsers
from utils import read_data, MyDataset
from torch.utils.data import DataLoader
from model import BertTextModel_encode_layer, BertTextModel_last_layer
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import sys

from transformers import BertTokenizer
import logging


logging.basicConfig(level=logging.DEBUG,
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')



def train(model, device, trainLoader, opt, epoch):
    model.train()
    loss_sum, count = 0, 0
    for batch_index, batch_con in enumerate(trainLoader):
        batch_con = [p.to(device) for p in batch_con]
        input_ids, attention_mask, token_type_ids, true_label = batch_con
        pred = model(input_ids, attention_mask, token_type_ids)

        opt.zero_grad()
        loss = loss_fn(pred, true_label)
        loss.backward()
        opt.step()
        loss_sum += loss
        count += 1

        if len(trainLoader) - batch_index <= len(trainLoader) % 1000 and count == len(trainLoader) % 1000:
            msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
            print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
            loss_sum, count = 0.0, 0

        if batch_index % 1000 == 999:
            msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
            print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
            loss_sum, count = 0.0, 0


def dev(model, device, devLoader, max_score):
    model.eval()
    all_true, all_pred = [], []
    for batch_con in tqdm(devLoader):
        batch_con = [p.to(device) for p in batch_con]
        input_ids, attention_mask, token_type_ids, true_label = batch_con
        pred = model(input_ids, attention_mask, token_type_ids)

        pred = torch.argmax(pred, dim=1)

        pred_label = pred.cpu().numpy().tolist()
        true_label = true_label.cpu().numpy().tolist()

        all_true.extend(true_label)
        all_pred.extend(pred_label)

    acc = accuracy_score(all_true, all_pred)
    print(f"dev acc:{acc:.4f}")

    if acc > max_score:
        max_score = acc
        torch.save(model.state_dict(), args.save_model_best)
        print(f"已保存最佳模型")
    return max_score


def run_predict(args, ls_predict_text, tokenizer):
    # jy: 基于配置参数, 选择指定的模型进行训练;
    if args.last_layer_only:
        model = BertTextModel_last_layer(args).to(device)
    else:
        model = BertTextModel_encode_layer(args).to(device)

    model_path = args.save_model_best
    logging.info("Loading model from: %s" % model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    logging.info("模型预测结果：")
    x = MyDataset(ls_predict_text, args.max_len, tokenizer)
    xDataloader = DataLoader(x, batch_size=len(ls_predict_text), shuffle=False)
    for batch_index, batch_con in enumerate(xDataloader):
        batch_con = [p.to(device) for p in batch_con]
        input_ids, attention_mask, token_type_ids = batch_con
        pred = model(input_ids, attention_mask, token_type_ids)

        results = torch.argmax(pred, dim=1)
        results = results.cpu().numpy().tolist()
        classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
        classification_dict = dict(zip(range(len(classification)), classification))
        for i in range(len(results)):
            logging.info(f"文本：{ls_predict_text[i]}\t预测的类别为：{classification_dict[results[i]]}")


def run_test(args, tokenizer):
    device = args.device

    logging.info("Loading test data from: %s" % args.test_file)
    ls_test_text, ls_test_label = read_data(args.test_file)
    # (ls_train_text, args.max_len, tokenizer, ls_train_label)
    test_dataset = MyDataset(ls_test_text, args.max_len, tokenizer, ls_test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # jy: 基于配置参数, 选择指定的模型进行训练;
    if args.last_layer_only:
        model = BertTextModel_last_layer(args).to(device)
    else:
        model = BertTextModel_encode_layer(args).to(device)

    model_path = args.save_model_best
    #model_path = args.save_model_last
    logging.info("Loading best model from: %s" % model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    logging.info("Begin to evaluate the model....")
    all_pred, all_true = [], []

    for batch_con in tqdm(test_dataloader):
        # jy: 将数据迁移至指定设备;
        batch_con = [p.to(device) for p in batch_con]
        input_ids, attention_mask, token_type_ids, labels = batch_con
        pred = model(input_ids, attention_mask, token_type_ids)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()

        all_pred.extend(pred)
        all_true.extend(labels)

    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, average="micro")
    recall = recall_score(all_true, all_pred, average="micro")
    f1 = f1_score(all_true, all_pred, average="micro")

    logging.info(f"test dataset accuracy:{accuracy:.4f}\tprecision:{precision:.4f}\trecall:{recall:.4f}\tf1:{f1:.4f}")


ls_predict_text = [
"我们一起去打篮球吧！",
"我喜欢踢足球！", 
"沈腾和马丽的新电影《独行月球》很好看", 
"昨天玩游戏，完了一整天",
"现在的高考都已经开始分科考试了。", 
"中方：佩洛西如赴台将致严重后果", 
"现在的股票基金趋势很不好"
]


if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = args.device if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    if args.is_do_predict_only:
        run_predict(args, ls_predict_text, tokenizer)
        # jy: 提前退出程序;
        sys.exit(1)

    if args.is_do_test_only:
        run_test(args, tokenizer)
        # jy: 提前退出程序;
        sys.exit(1)


    # jy: 加载训练集;
    ls_train_text, ls_train_label = read_data(args.train_file)
    trainData = MyDataset(ls_train_text, args.max_len, tokenizer, ls_train_label)
    trainLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True)

    # jy: 加载开发集;
    ls_dev_text, ls_dev_label = read_data(args.dev_file)
    devData = MyDataset(ls_dev_text, args.max_len, tokenizer, ls_dev_label)
    devLoader = DataLoader(devData, batch_size=args.batch_size, shuffle=True)


    # jy: 基于配置参数, 选择指定的模型进行训练;
    if args.last_layer_only:
        model = BertTextModel_last_layer(args).to(device)
    else:
        model = BertTextModel_encode_layer(args).to(device)


    opt = AdamW(model.parameters(), lr=args.learn_rate)
    loss_fn = CrossEntropyLoss()

    max_score = float("-inf")
    for epoch in range(args.epochs):
        # jy: 训练模型;
        train(model, device, trainLoader, opt, epoch)
        # jy: 开发集上校验并判断是否保存最佳模型;
        max_score = dev(model, device, devLoader, max_score)

    model.eval()
    torch.save(model.state_dict(), args.save_model_last)

    end = time.time()
    print(f"运行时间：{(end-start)/60%60:.4f} min")
