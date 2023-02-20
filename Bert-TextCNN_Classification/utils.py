from torch.utils.data import Dataset
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
    def __init__(self, ls_text, max_len, tokenizer, ls_label=None):
        self.ls_text = ls_text
        self.ls_label = ls_label
        self.max_len = max_len
        self.tokenizer = tokenizer 

    def __getitem__(self, index):
        text = self.ls_text[index]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        # jy: 以下直接调用 tokenizer 类直接进行 tokenize 处理;
        encoded_pair = self.tokenizer(text,
                                      # Pad to max_length
                                      padding='max_length',
                                      # Truncate to max_length
                                      truncation=True, 
                                      max_length=self.max_len,
                                      # Return torch.Tensor objects
                                      return_tensors='pt')  

        # tensor of token ids  torch.Size([max_len])
        token_ids = encoded_pair['input_ids'].squeeze(0)
        # binary tensor with "0" for padded values and "1" for the other values  torch.Size([max_len])
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens  torch.Size([max_len])
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

        # True if the dataset has labels
        if self.ls_label:  
            label = int(self.ls_label[index])
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids


    def __len__(self):
        # jy: 获取文本的长度
        return len(self.ls_text)


