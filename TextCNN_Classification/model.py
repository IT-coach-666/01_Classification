import torch.nn as nn
import torch


class Block(nn.Module):
    def __init__(self, kernel_s, embeddin_num, max_len, hidden_num):
        super().__init__()
        # jy: self.cnn 如: Conv2d(1, 2, kernel_size=(2, 50), stride=(1, 1))
        self.cnn = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(kernel_s, embeddin_num))
        # jy: 激活函数;
        self.act = nn.ReLU()
        # jy: MaxPool1d(kernel_size=36, stride=36, padding=0, dilation=1, ceil_mode=False)
        self.mxp = nn.MaxPool1d(kernel_size=(max_len - kernel_s + 1))

    def forward(self, batch_emb):
        """
        输入参数:
        batch_emb: 维度为 [batch, in_channel, max_len, emb_num],
                   如: [batch_size, 1, max_len, embedding_num]

        返回结果: 维度为 torch.Size([batch_size, 2])
        """
        # jy: batch_emb 维度如: torch.Size([32, 1, 38, 50])
        #     经 self.cnn 操作后得到的 c 的维度如: torch.Size([32, 2, 37, 1])
        c = self.cnn(batch_emb)
        # jy: 应用激活函数, 不改变维度;
        a = self.act(c)
        # jy: 经 squeeze(dim=-1) 处理后的 a 的维度: torch.Size([32, 2, 37])
        a = a.squeeze(dim=-1)
        # jy: 经 self.mxp 处理后得到的 m 的维度如: torch.Size([32, 2, 1])
        m = self.mxp(a)
        # jy: 最终返回的 m 的维度如: torch.Size([32, 2])
        m = m.squeeze(dim=-1)
        return m


class TextCNNModel(nn.Module):
    def __init__(self, emb_matrix, max_len, class_num, hidden_num):
        super().__init__()
        # jy: 词表对应的向量, 如: Embedding(3443, 50); 其中 3443 表示词表大小, 50 为词向量维度;
        self.emb_matrix = emb_matrix
        # jy: 获取向量维度;
        self.emb_num = emb_matrix.weight.shape[1]

        # jy: 初始化 Block 类;
        self.block1 = Block(2, self.emb_num, max_len, hidden_num)
        self.block2 = Block(3, self.emb_num, max_len, hidden_num)
        self.block3 = Block(4, self.emb_num, max_len, hidden_num)


        # jy: Linear(in_features=6, out_features=10, bias=True)
        self.classifier = nn.Linear(hidden_num * 3, class_num) 
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_idx): 
        """
        传入参数: 
        batch_idx: 维度为 [batch_size, 1, max_len]

        返回结果: 维度为 [batch_size, class_num] 的向量;
        """
        # jy: 获取一个 batch 序列对应的向量, 维度为: [batch_size, 1, max_len, embedding_num]
        batch_emb = self.emb_matrix(batch_idx)  
        # jy: 经过 block 处理后的结果向量维度均为: [batch_size, 2]
        b1_result = self.block1(batch_emb)
        b2_result = self.block2(batch_emb)
        b3_result = self.block3(batch_emb) 

        # jy: 拼接, 得到的 feature 维度为: [batch_size, 6]
        feature = torch.cat([b1_result, b2_result, b3_result], dim=1) 
        # jy: 经线性分类器后, 得到的结果向量维度为: [batch_size, class_num]
        pre = self.classifier(feature)

        return pre

