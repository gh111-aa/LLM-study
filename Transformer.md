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

<img width="1085" height="775" alt="c96073254a445486ed8aed22460f9b01" src="https://github.com/user-attachments/assets/e7cd15be-2bc0-449f-873d-456749133b2a" />

### embedding


#### embedding

- 首先对于词进行切割
- 补上[CLS]和[SEP]，分隔两个句子
<img width="994" height="614" alt="0f9ed258a590b73e6d69454f263ed461" src="https://github.com/user-attachments/assets/3d8bf46b-6df3-4c6e-a5d5-0a33ae7030e4" />

<img width="1009" height="590" alt="5b93bf0b1d947183ae08ce65b4dc9183" src="https://github.com/user-attachments/assets/c58c0125-4be2-4afa-837c-a18148e64af3" />

<img width="990" height="280" alt="0c517d4ef0d7f3c0350a453e02e0e913" src="https://github.com/user-attachments/assets/0efcd959-e15a-4c3b-82e3-4a2ea9e3e3dc" />

*因为注意力矩阵的产生是受每个词的影响，很有可能其他词的出现会影响另一个词对于所有词的关联度，在多层训练的影响下，[SEP]的数值会慢慢的达到一个分隔符的作用，即让前后两个句子之间的关联度减弱*

- token embedding
- segment embedding

*使用位置嵌入＋句子分隔其实不能代替segment embedding，因为transformer的自注意力机制会让第二句话的字来思索它和第一句话的关系，如果第二句话很短，和第一句话关系不大，那么这就会导致这个字理解错误*

- position embedding

#### transformer encoder

#### 预训练
- MLM（mask language model）

随即掩码的机制来调整模型参数

<img width="1024" height="323" alt="d3bbd39325e0262e517aa46206c11611" src="https://github.com/user-attachments/assets/3770cb5f-cc82-44e5-95c0-0ed23cc2040b" />

- NSP

<img width="988" height="491" alt="f29f1a244625b1bd01b83bcad7be22c0" src="https://github.com/user-attachments/assets/51f9552f-85ea-4252-8185-461f2665f4e7" />

<img width="928" height="538" alt="efe6f1cc7e359908159fca7f07b70ced" src="https://github.com/user-attachments/assets/f6ae573b-3a43-49d1-b7a7-7b2b0c27d7d7" />

<img width="951" height="475" alt="90d046ae0135016270c3b969865551d3" src="https://github.com/user-attachments/assets/b365c4ff-67b0-4ed7-9fd2-60403261cb90" />










