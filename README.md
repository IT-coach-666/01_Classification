# Bert_Classification
基于 bert 模型进行 fine-tune, 实现中文文本分类（10 类）  

模型都未进行调参，未能使模型的准确率达到最高  


BERT的总体预训练和微调程序。除了输出层之外，在预训练和微调中使用相同的架构。相同的预训  
练模型参数用于初始化不同下游任务的模型。在微调期间，将微调所有参数。  
[CLS] 是一个特殊的符号添加在每个输入示例的前面  
[SEP] 是一个特殊的分隔符标记（例如，分隔问题/答案）  


Bert 模型的输入  
BERT 的输入可以包含一个句子对 (句子 A 和句子 B)，也可以是单个句子。同时 BERT 增加了一些有特殊作用的标志位：  
[CLS] 标志放在第一个句子的首位，经过 BERT 得到的的表征向量 C 可以用于后续的分类任务。  
[SEP] 标志用于分开两个输入句子，例如输入句子 A 和 B，要在句子 A，B 后面增加 [SEP] 标志。  
[MASK] 标志用于遮盖句子中的一些单词，将单词用 [MASK] 遮盖之后，再利用 BERT 输出的 [MASK] 向量预测单词是什么。  


Bert 模型的 Embedding 模块
BERT 得到要输入的句子后，要将句子的单词转成 Embedding，Embedding 用 E 表示。
与 transformer 不同，BERT 的输入 Embedding 由三个部分相加得到：Token Embedding，Segment Embedding，position Embedding。
Token Embedding：单词的 Embedding，例如 [CLS] dog 等，通过训练学习得到。
Segment Embedding：用于区分每一个单词属于句子 A 还是句子 B，如果只输入一个句子就只使用 EA，通过训练学习得到。
position Embedding：编码单词出现的位置，与 transformer 使用固定的公式计算不同，BERT 的 position Embedding 也是通过学习得到的，在 BERT 中，假设句子最长为 512。




项目数据集
数据集使用 THUCNews 中的 train.txt、test.txt、dev.txt，为十分类问题。
其中训练集一共有 180000 条，验证集一共有 10000 条，测试集一共有 10000 条。
其类别为 finance、realty、stocks、education、science、society、politics、sports、game、entertainment 这十个类别。

模型训练
`python main.py`

模型预测
`python predict.py`



训练自己的数据集
train.txt、dev.txt、test.txt 的数据格式：文本\t标签（数字表示）

体验2D巅峰 倚天屠龙记十大创新概览\t8
60年铁树开花形状似玉米芯(组图)\t5

class.txt：标签类别（文本）
修改内容：
在配置文件中修改长度、类别数、预训练模型地址
parser.add_argument("--bert_pred", type=str, default="./bert-base-chinese")
parser.add_argument("--class_num", type=int, default=10)
parser.add_argument("--max_len", type=int, default=38)


