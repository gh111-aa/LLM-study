# LLM
## 一、算法与深度学习基础
### 传统机器学习核心：
#### LR（逻辑回归）：

虽然是个回归，但是其实是个二分类模型。Logistic分布如下：
<img width="996" height="460" alt="2e1394c179098b4820c35d3c9caf7625" src="https://github.com/user-attachments/assets/681b9e7c-4c81-4500-8202-5127ee22c029" />

假设数据集中的两个标签可以用一条线划分开：wx+b>0,那么我们想知道P(y=1|x)条件概率，最好的是单位跃阶概率函数，但是因为不可导，所以使用logistic分布的特例sigmoid函数进行概率的近似：

<img width="750" height="320" alt="7ca85415055495788115a9527e15c732" src="https://github.com/user-attachments/assets/e1f6c2e0-a24e-4977-a961-87cc8c9f1124" />

“近似”过程解释如下：（假设了）

<img width="974" height="554" alt="ccbc45d6946930fdc874cb1423bb97c2" src="https://github.com/user-attachments/assets/e8d1d486-551b-49b1-9c47-f7a55e01a5f5" />


<img width="980" height="449" alt="640329e32de94d90e86817431245f221" src="https://github.com/user-attachments/assets/01cb1f84-fdb6-4c27-8d88-6128a9691fff" />

所以逻辑回归的好处是：

<img width="1115" height="240" alt="9085ffa3a8cc9dd08980152a6f420a65" src="https://github.com/user-attachments/assets/814de788-d452-4936-a89c-b16993adc35d" />

极大似然估计参数w,b

<img width="815" height="375" alt="af5e3630eca7754018c5c2907b223162" src="https://github.com/user-attachments/assets/dd7855a3-c3ca-49de-ad2a-30c288543779" />

<img width="590" height="188" alt="e3816024694e045140958e031fe7c237" src="https://github.com/user-attachments/assets/6c6411a0-d3bb-430c-b923-fcc30395d5d4" />

根据机器学习的思想引出损失函数（其实是变体）：

损失函数可以进行正则化（L1正则和L2正则），L1正则如下：

<img width="930" height="736" alt="93da9b17c06e094159f23f5bb8f60a0a" src="https://github.com/user-attachments/assets/c2ef43e5-6f01-4c98-b014-2de018173c9b" />

L2正则如下：

<img width="848" height="720" alt="c12956617cdc4e6d63778b4f29380b00" src="https://github.com/user-attachments/assets/f89a004b-11f2-4a1e-9964-e48f1731af52" />

<img width="903" height="84" alt="4c8052f1486e4b156bba54934112d9d2" src="https://github.com/user-attachments/assets/d087c69b-4c36-41c4-bddc-7ad834a1bbb1" />

L1 正则化增加了所有权重 w 参数的绝对值之和逼迫更多 w 为零

L2 正则化中增加所有权重 w 参数的平方之和，逼迫所有 w 尽可能趋向零但不为零（L2 的导数趋于零）。

<img width="906" height="453" alt="864db01dec9561b5da76dd52c6647679" src="https://github.com/user-attachments/assets/f93bb1ae-0c55-47a0-a929-4c7a3d376dca" />

如果还像原来只优化 f 的情况下，那可能得到一组解比较复杂，使得正则项 ||w|| 比较大，那么 h 就不是最优的，因此可以看出加正则项能让解更加简单，符合奥卡姆剃刀理论，同时也比较符合在偏差和方差（方差表示模型的复杂度）分析中，通过降低模型复杂度，得到更小的泛化误差，降低过拟合程度。

L1 正则化就是在 loss function 后边所加正则项为 L1 范数，加上 L1 范数容易得到稀疏解（0 比较多）。L2 正则化就是 loss function 后边所加正则项为 L2 范数的平方，加上 L2 正则相比于 L1 正则来说，得到的解比较平滑（不是稀疏），但是同样能够保证解中接近于 0（但不是等于 0，所以相对平滑）的维度比较多，降低模型的复杂度。

文章参考知乎：https://zhuanlan.zhihu.com/p/74874291

#### K-means聚类
**步骤一** 初始化：选择K个初始聚类中心

算法开始时，需要随机选择K个数据点作为初始的聚类中心，这个数据中心的选择会对聚类结果有影响，所以通常会采用一些启发式方法来选择较好的聚类中心，如K-means++算法

选择聚类数K：

<img width="1320" height="246" alt="74234cdff1f434b3e7bea515211cb6d2" src="https://github.com/user-attachments/assets/9746b9b6-cc31-4907-8114-c5bcbf18280d" />

选择聚类中心：

<img width="1360" height="228" alt="ace671938fa846d60aa411ffc6a46a50" src="https://github.com/user-attachments/assets/29390d0b-3af6-4117-bd4f-c13dd6b3e2c8" />

**步骤二** 分配：计算数据点到聚类中心距离，分配聚类点

这涉及到距离的定义，欧氏距离、曼哈顿距离、余弦相似度等等

**步骤三** 更新：重新计算每个聚类的中心

**步骤四** 迭代：重复二三步，直到满足终止条件

聚类中心不再显著变化

达到最大迭代次数

算法的加速方法：

<img width="1345" height="290" alt="2132bdebe26947f3400794d849d685de" src="https://github.com/user-attachments/assets/7068387a-aaa0-4539-9ed0-736f8529f633" />

KD法本质上是一种更改数据结构的方法

KNN紧邻算法，本质上是在KD法更改数据结构后在新的数据结构上搜索最近的方法（可以剪枝）

KNN算法参考知乎：https://zhuanlan.zhihu.com/p/23966698

#### 梯度下降法

一个点的偏导数向量是定义域里该点上升最快的方向。

<img width="906" height="251" alt="a9636d4c8775e89953d3258a3528f597" src="https://github.com/user-attachments/assets/a69db230-dc25-4e15-b37e-51f983502c4c" />

- 随机梯度下降

<img width="935" height="358" alt="3c623108ba53f5e16746212dfcdf1d8e" src="https://github.com/user-attachments/assets/9f633742-ca57-409e-885d-314c8774cde7" />

- 批梯度下降

就是总误差函数直接偏导来迭代

#### 常见loss

- 交叉熵：

<img width="935" height="476" alt="6f5c7d8f8bb7dde5456042a8ae462dcc" src="https://github.com/user-attachments/assets/a94474ec-26ed-470d-a1b0-5d72de866ca9" />

<img width="954" height="309" alt="a83c993e5703666eaa9b32a8d6cfdc5b" src="https://github.com/user-attachments/assets/631e262b-acef-44dd-b278-032e2d7ab76c" />

可以联想LR逻辑回归里面求解P(y|x)所使用的极大似然法，导出的损失函数也是交叉熵

- MSE

均方误差容易被奇异值影响

#### Adam

- mini batch、batch size、Mini-Batch梯度下降

- 指数加权平均

<img width="944" height="360" alt="861883cb2ad2654956a11ee6e3e8fc11" src="https://github.com/user-attachments/assets/6394176b-d454-4ade-be23-13a382dbc88c" />

误差纠正：

<img width="1050" height="506" alt="aac4681a115892811980fdd93802157c" src="https://github.com/user-attachments/assets/e6ccf90f-513d-48c1-91e4-f8ebc04b490c" />

<img width="1034" height="256" alt="c974bb278f1c747ed14d5c232ef57015" src="https://github.com/user-attachments/assets/426d35c0-9152-40fa-bcfe-b02436180e4e" />

- Momentum梯度下降法（针对batch）

在一个epoch中每次选择batch样本对应的损失函数进行迭代，这样的区别在于迭代方向的变化。
比随机梯度下降稳定，比批梯度下降方便。

- RMSprop（自适应控制步长即学习率）

<img width="1044" height="561" alt="1bce42239013ba6de129df272c01bfb9" src="https://github.com/user-attachments/assets/6a1526f6-8491-4c96-b58a-3339d8347764" />

<img width="965" height="508" alt="15ca3cf36921539d089b6b458dd300e2" src="https://github.com/user-attachments/assets/ce9ca2f4-1011-4585-a218-4d7fc671f7f0" />

里面的Sdw要求是梯度平方的指数加权平均是为了刻画梯度序列的“大小”（平方代表向量长度），所以这样设计学习率是为了在梯度陡峭的时候步长缩小，梯度平缓“走不动”的时候能够加大步长。

- Adam Optimizer（结合Momentum和RMSprop）

<img width="921" height="580" alt="dd43fb08c56da732b674d1c435295023" src="https://github.com/user-attachments/assets/e7d133f5-9933-4539-afc9-35cbeef912c5" />

<img width="946" height="674" alt="25c0d73b5c2b1b89f424df5c0741d9e8" src="https://github.com/user-attachments/assets/408eb06d-fe24-43b1-90c2-6a2dcd9e9e42" />

把a/根号s看成一个整体，这是针对步长的。
























《动手学深度学习》：https://zh.d2l.ai/index.html



