# Transformer
## 架构理解
<img width="610" height="845" alt="66dd24ea645f645526f46d7e83cd145c" src="https://github.com/user-attachments/assets/fd3d2321-68ab-4a70-886d-48ecba85db78" />

### encoder部分：本质上是利用“相关度、注意力”来编码、提取每个位置的字的内在特征
- self-attention

- Multi-head attention

- add+norm（残差网络＋层正则）

add可以防止神经网络退化（层数越多反倒拟合效果更差）
norm可以使训练会更加迅速

- feed forward(全连接)

### decoder部分：
*mask多头注意力层分析已经生成的M个字（包含<sos>）的内在因果结构，提取出第M个字的查询向量，该向量考虑了已经生成的M个字的因果关系，来查询encoder部分的V键向量，寻找encoder输入部分的相关关系，生成注意力矩阵，来查询V值向量，这样的行为既保留了翻译产生的句子的逻辑完整性，又能够翻译encoder部分的句子。*


*如果这个transformer进行的是翻译工作，encoder输入的是中文，要求decoder输出英文，那么当M等于1的时候，只有一个<sos>，这个时候查询矩阵只查第一个字的因果关系，（子注意力为1），乘以值向量相当于恒等变换，输出之后通过linear查询英文，按理说应该在英文的数据库里面进行查询，所以encoder和decoder的语类可能不一样，翻译工作使用这样的encoder-decoder是合理的*


- mask操作

因果掩码

填充掩码

- linear

# transformer三大变体

预训练、后训练、强化学习简单介绍：https://zhuanlan.zhihu.com/p/32208170972

## only-encoder DERT










