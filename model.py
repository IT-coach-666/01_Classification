import torch.nn as nn
from transformers import BertModel
import torch
from transformers import BertPreTrainedModel


class MyModel(BertPreTrainedModel):
    def __init__(self, config):
        super(MyModel, self).__init__(config)

        # jy: 加载 bert 预训练模型
        self.bert = BertModel(config) 

        # jy: 让 bert 模型进行微调(参数在训练过程中变化), 默认情况
        #     下 param.requires_grad 即为 True, 故以下不需要特意设置;
        """
        for param in self.bert.parameters():
            param.requires_grad = True
            #print(param.requires_grad)
        """

        # jy: 全连接层 config.hidden_size 为 768, config.num_lables 为 10;
        self.linear = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None):
        """
        bert 的输出结果:
        last_hidden_state: 维度为 (batch_size, sequence_length, hidden_size), 是模型最后一层输出的隐藏状态
        pooler_output: 维度为 (batch_size, hidden_size), 是序列的第一个 token (classification token) 的最后
                       一层的隐藏状态, 它由线性层和 Tanh 激活函数进一步处理得到; 通常用于句子分类, 至于是使
                       用这个表示, 还是使用整个输入序列的隐藏状态序列的平均化或池化, 视情况而定.
        hidden_states: 可选项, 指定 config.output_hidden_states=True 时会输出, 是一个元组, 元组的第一个元素
                       是 embedding, 其余元素是各层的输出, 每个元素的向量维度同 last_hidden_state 的维度
        attentions: 可选项, 指定 config.output_attentions=True 时输出, 也是一个元组, 元素是每一层的注意力
                    权重, 用于计算 self-attention heads 的加权平均值
        cross_attentions: 维度为 (batch_size, num_heads,encoder_sequence_length, embed_size_per_head)
        """
        hidden_out = self.bert(input_ids, attention_mask=attention_mask,
                               output_hidden_states=False)  # 控制是否输出所有encoder层的结果

        # shape (batch_size, hidden_size) 
        #import pdb; pdb.set_trace()
        pred = self.linear(hidden_out[1])
        # 返回预测结果
        return pred


