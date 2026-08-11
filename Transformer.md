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
*mask多头注意力层分析已经生成的M个字的mask后的内在结构*

- mask操作
因果掩码
填充掩码














