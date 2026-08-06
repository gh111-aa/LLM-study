# LLM
## 一、算法与深度学习基础
### 传统机器学习核心：
- LR（逻辑回归）：

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













