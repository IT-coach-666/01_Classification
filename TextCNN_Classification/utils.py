import os
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import pickle as pkl

from pysenal import read_lines_lazy


def read_data(f_name):
    # 得到所有文本、所有标签、句子的最大长度
    ls_text, ls_label = [], []
    for line in read_lines_lazy(f_name):
        line = line.strip()
        if not line:
            continue

        text, label = line.split("\t")
        ls_text.append(text)
        ls_label.append(label)
    return ls_text, ls_label


def built_curpus(args, ls_text, embedding_num):
    """
    注意, 该函数目前只适合于中文文本, 因为当前的处理逻辑是将每个字符当做一个 token;
    更加正规的方式是使用已有的 tokenize 工具进行 tokenize 处理后再基于相应的结果构
    造词表和初始化词向量;
    """
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for text in ls_text:
        # jy: 针对中文文本, 可以将每个字符当做一个 token 进行相关的词表构建
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    # jy: 基于构建的词表初始化相应的词向量;
    embedding = nn.Embedding(len(word_2_index), embedding_num)
    # jy: 将构建的词表和初始化的词向量本地化保存;
    pkl.dump([word_2_index, embedding], open(args.data_pkl, "wb"))
    return word_2_index, embedding


class TextDataset(Dataset):
    def __init__(self, ls_text, word_2_index, max_len, ls_label=None):
        self.ls_text = ls_text
        self.ls_label = ls_label
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.ls_text[index][:self.max_len]

        text_idx = [self.word_2_index.get(i, 1) for i in text]
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))

        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)
        if self.ls_label:
            label = int(self.ls_label[index])
            return text_idx, label
        else:
            return text_idx

    def __len__(self):
        return len(self.ls_text)

