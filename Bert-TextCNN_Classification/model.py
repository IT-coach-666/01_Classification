from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class TextCnnModel(nn.Module):
    def __init__(self, args):
        super(TextCnnModel, self).__init__()
        self.args = args
        # jy: self.args.num_filters: 2
        #     self.args.filter_sizes: [2, 3, 4]
        self.num_filter_total = self.args.num_filters * len(self.args.filter_sizes)
        # jy: self.Weight 为: Linear(in_features=6, out_features=10, bias=False)
        self.Weight = nn.Linear(self.num_filter_total, self.args.class_num, bias=False)
        # jy: 维度为 torch.Size([10]) 的 <class 'torch.nn.parameter.Parameter'>
        self.bias = nn.Parameter(torch.ones([self.args.class_num]))
        # jy: self.filter_list 为:
        """
        ModuleList(
          (0): Conv2d(1, 2, kernel_size=(2, 768), stride=(1, 1))
          (1): Conv2d(1, 2, kernel_size=(3, 768), stride=(1, 1))
          (2): Conv2d(1, 2, kernel_size=(4, 768), stride=(1, 1))
        )
        """
        self.filter_list = nn.ModuleList([
            # jy: self.args.num_filters: 2
            #     self.args.hidden_size: 768
            #     self.args.filter_sizes: [2, 3, 4]
            nn.Conv2d(1, self.args.num_filters, kernel_size=(size, self.args.hidden_size))
            for size in self.args.filter_sizes])

    def forward(self, x):
        """
        传入参数: 
        x: 维度为 [batch_size, 12, hidden] 的向量;
        """
        # jy: 原始的 x 维度为: [batch_size, 12, hidden]
        #     经过转换后的 x 维度为: [batch_size, channel=1, 12, hidden]
        x = x.unsqueeze(1) 

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            # jy: 转换后得到的 out 维度如: [batch_size, channel=2, 12-kernel_size[0]+1, 1]
            #     x 维度如: [batch_size, channel=1, 12, hidden]
            #     conv 如:  Conv2d(1, 2, kernel_size=(2, 768), stride=(1, 1))
            #     经转换后得到的维度如: torch.Size([4, 2, 11, 1])
            out = F.relu(conv(x)) 
            # jy: 得到的 maxPool 如: 
            #     MaxPool2d(kernel_size=(11, 1), stride=(11, 1), padding=0, dilation=1, ceil_mode=False)
            maxPool = nn.MaxPool2d(
                kernel_size=(self.args.encode_layer - self.args.filter_sizes[i] + 1, 1)
            )
            # jy: maxPool(out) 处理后的维度为: [batch_size, channel=2, 1, 1]
            #     经调整后最终的 pooled 的维度为: [batch_size, h=1, w=1, channel=2]
            pooled = maxPool(out).permute(0, 3, 2, 1) 
            pooled_outputs.append(pooled)

        # jy: pooled_outputs 为含三个元素的列表, 列表中的元素是维度为 [batch_size, h=1, w=1, channel=2] 的向量;
        #     经以下处理后, h_pool 的维度为: [batch_size, h=1, w=1, channel=2 * 3]
        h_pool = torch.cat(pooled_outputs, len(self.args.filter_sizes))
        # jy: 经以下处理后, h_pool_flat 的维度为: [batch_size, 6]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
        # jy: 经转换后, 得到的 output 维度为: [batch_size, class_num]
        output = self.Weight(h_pool_flat) + self.bias 

        return output


class BertTextModel_encode_layer(nn.Module):
    def __init__(self, args):
        super(BertTextModel_encode_layer, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(self.args.model_name_or_path)

        # jy: param.requires_grad 不需手动设置默认即为 True;
        """
        for param in self.bert.parameters():
            param.requires_grad = True
            print(param.requires_grad)
        """
        # jy: self.args.hidden_size: 768
        #     self.args.class_num: 10
        self.linear = nn.Linear(self.args.hidden_size, self.args.class_num)
        # jy: self.textCnn 为: 
        """
        TextCnnModel(
          (Weight): Linear(in_features=6, out_features=10, bias=False)
          (filter_list): ModuleList(
             (0): Conv2d(1, 2, kernel_size=(2, 768), stride=(1, 1))
             (1): Conv2d(1, 2, kernel_size=(3, 768), stride=(1, 1))
             (2): Conv2d(1, 2, kernel_size=(4, 768), stride=(1, 1))
           )
        )
        """
        self.textCnn = TextCnnModel(self.args)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            # 确保 hidden_states 的输出有值
                            output_hidden_states=True)
        # jy: 取每一层 encode 出来的向量, 得到的 hidden_states 对应一个 13 元组, 元组的第一个元素
        #     是 embedding 层, 不需要(以下代码逻辑会从元组的第二个元素开始获取); 
        #     元组中的元素的维度均为: [batch_size, seq_max_len, hidden_size] 
        hidden_states = outputs.hidden_states 
        # jy: hidden_states[1] 的维度为: [batch_size, seq_max_len, hidden_size]
        #     hidden_states[1][:, 0, :] 的维度为: [batch_size, hidden_size]
        #     hidden_states[1][:, 0, :].unsqueeze(1) 的维度为: [batch_size, 1, hidden]
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1) 
        # jy: 将每一层的第一个 token (cls 向量) 提取出来, 拼在一起当作 textCnn 的输入
        for i in range(2, 13):
            # jy: 每循环一次, cls_embeddings 向量中的第二个维度值就不断加 1;
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # jy: 经过以上循环后, cls_embeddings 向量维度为: [batch_size, 12, hidden]
        pred = self.textCnn(cls_embeddings)
        return pred


class BertTextModel_last_layer(nn.Module):
    def __init__(self, args):
        super(BertTextModel_last_layer, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(self.args.model_name_or_path)

        # jy: param.requires_grad 不需手动设置默认即为 True;
        """
        for param in self.bert.parameters():
            param.requires_grad = True
            print(param.requires_grad)
        """

        # TextCNN
        """
        ModuleList(
          (0): Conv2d(1, 2, kernel_size=(2, 768), stride=(1, 1))
          (1): Conv2d(1, 2, kernel_size=(3, 768), stride=(1, 1))
          (2): Conv2d(1, 2, kernel_size=(4, 768), stride=(1, 1))
        )
        """
        self.convs = nn.ModuleList(
            # jy: self.args.num_filters:  2
            #     self.args.hidden_size:  768
            #     self.args.filter_sizes: [2, 3, 4]
            [nn.Conv2d(in_channels=1, out_channels=self.args.num_filters, kernel_size=(k, self.args.hidden_size),)
             for k in self.args.filter_sizes]
        )
        self.dropout = nn.Dropout(self.args.dropout)
        # jy: self.fc 为: Linear(in_features=6, out_features=10, bias=True)
        #     self.args.num_filters:  2
        #     self.args.filter_sizes: [2, 3, 4]
        #     self.args.class_num:    10
        self.fc = nn.Linear(self.args.num_filters * len(self.args.filter_sizes), self.args.class_num)

    def conv_pool(self, x, conv):
        """
        传入参数如:
        x: 维度为 [batch_size, 1, max_len, hidden_size] 的向量;
        conv: 以下三种中的一种
              Conv2d(1, 2, kernel_size=(2, 768), stride=(1, 1))
              Conv2d(1, 2, kernel_size=(3, 768), stride=(1, 1))
              Conv2d(1, 2, kernel_size=(4, 768), stride=(1, 1))

        返回结果的向量维度为: [batch_size, out_channels]
        """ 
        # jy: 转换后的 x 的维度为: [batch_size, out_channels, x.shape[1] - conv.kernel_size[0] + 1, 1]
        #     如: conv 为: Conv2d(1, 2, kernel_size=(2, 768), stride=(1, 1))
        #         x 的维度为: torch.Size([4, 1, 38, 768])
        #         转换后的 x 的维度是: torch.Size([4, 2, 37, 1])
        x = conv(x)  
        # jy: 激活函数, 不改变 x 的维度;
        x = F.relu(x)
        # jy: 经转换后 x 的维度为: [batch_size, out_channels, x.shape[1] - conv.kernel_size[0] + 1]
        #     如原始 x 的维度为: torch.Size([4, 2, 37, 1]), 经转换后为: torch.Size([4, 2, 37])
        x = x.squeeze(3) 
        # jy: x 维度为 torch.Size([4, 2, 37]) 时, size 得到的是 37;
        size = x.size(2)
        # jy: 转换后 x 的维度为: [batch+size, out_channels, 1]   
        # jy: 如: 原始 x 的维度: torch.Size([4, 2, 37]), 经转换后得到的 x 的维度如: torch.Size([4, 2, 1])
        x = F.max_pool1d(x, size) 
        # jy: 转换后 x 的维度为: [batch_size, out_channels]
        #     如: 原 x 的维度为 torch.Size([4, 2, 1]), 转换后的 x 维度为: torch.Size([4, 2])
        x = x.squeeze(2) 
        return x


    def forward(self, input_ids, attention_mask, token_type_ids):
        """

        返回结果的向量维度为: [batch_size, class_num]
        """
        # jy: 均为维度为: [batch_size, max_len] 的向量;
        #input_ids, attention_mask, token_type_ids = x[0], x[1], x[2]
        hidden_out = self.bert(input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               output_hidden_states=False)
        # jy: hidden_out.last_hidden_state 的维度为: [batch_size, max_len, hidden_size]
        #     经过以下转换后得到的 out 的维度为: [batch_size, 1, max_len, hidden_size]
        out = hidden_out.last_hidden_state.unsqueeze(1) 
        # jy: 经过以下转换后得到的 out 的维度为: [batch_size, self.args.num_filters * len(self.args.filter_sizes]
        out = torch.cat([self.conv_pool(out, conv) for conv in self.convs], 1) 
        # jy: 经过 dropout 处理, 维度不变;
        out = self.dropout(out)
        # jy: 经过 self.fc 处理后, 最终 out 的维度为: [batch_size, class_num]
        out = self.fc(out)
        return out


