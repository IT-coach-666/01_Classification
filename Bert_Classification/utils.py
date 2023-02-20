from torch.utils.data import Dataset, DataLoader
import torch
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


class MyDataset(Dataset):
    def __init__(self, ls_text, ls_label, max_length, tokenizer):
        self.ls_text = ls_text
        self.ls_label = ls_label
        self.max_len = max_length
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        # jy: 取出一条数据并截断, 得到的 text 如: 'A股难有大行情 期待十月突围' 
        text = self.ls_text[index][:self.max_len]
        # jy: label 如 "2"
        label = self.ls_label[index]

        # jy: 以下是调用 self.tokenizer 的 tokenize 先将 text 文本进行 tokenize, 再将 token 转换为
        #     对应的 token_id, 该方式为了最终转换得到的 token_id 中包含 "[CLS]" 对应的 token_id, 需要
        #     提前中 token 文本列表中加入该标签; 
        #     如果直接调用 self.tokenizer 类对 text 文本进行处理, 则不需要为 text 文本得到的 token 列表
        #     添加 "[CLS]", 内部处理过程中即会自动进行该处理, 使得得到的 input_ids 中第一个 id 值即为 101;
        # jy: 分词, 得到的 text_id 如:
        #     ['a', '股', '难', '有', '大', '行', '情', '期', '待', '十', '月', '突', '围']
        text_id = self.tokenizer.tokenize(text)
        # jy: 在 text_id 前加上起始标志
        text_id = ["[CLS]"] + text_id

        # jy: 编码, 得到的 token_id 如:
        #     [101, 143, 5500, 7410, 3300, 1920, 6121, 2658, 3309, 2521, 1282, 3299, 4960, 1741]
        token_id = self.tokenizer.convert_tokens_to_ids(text_id)
        # jy: 掩码, 得到的 mask 结果如(长度为 max_length + 2):
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        mask = [1] * len(token_id) + [0] * (self.max_len + 2 - len(token_id))
        # jy: 将 token_id 补全, 使得长度与 mask 的长度一致
        token_id = token_id + [0] * (self.max_len + 2 - len(token_id))
        # jy: str -> int
        label = int(label)

        # jy: 转化成 tensor; 该过程其实可以放到上一层再转换; 
        token_id = torch.tensor(token_id)
        mask = torch.tensor(mask)
        label = torch.tensor(label)

        return (token_id, mask), label

    def __len__(self):
        # 得到文本的长度
        return len(self.ls_text)

