**PLSA(Probabilistic Latent Semantic Analysis)**概率潜在语义分析, 与LSA相比, 有着更坚实的数学基础.

## 意向模型

PLSA的核心思想即是意向模型(Aspect Model). 对于**主题**, 将其对应于**隐变量**$$z\in{Z}=\{z_1,\cdots,z_K\}$$, 并与词汇$$w\in{W}=\{w_1,\cdots,w_M\}$$和文档$$d\in{D}=\{d_1,\cdots,d_N\}$$联系起来, 组成统计模型. 同时我们认为每篇文本的每个单词生成过程如下:

- 一篇文本对应一个$$K$$面的骰子, 选出一个主题
- 一个主题对应一个$$M$$面的骰子, 选出一个单词

因此有以下关系:

- $$d_i$$和$$w_j$$是相对独立的
- $$w_j$$产生只依赖于$$z_k$$, 不依赖与$$d_i$$

则模型可以表示为词与文档的联合概率:

$$P(d_i,w_j)=P(d_i)P(w_j|d_i) \tag{1}$$

$$P(w_j|d_i)=\sum\limits_{k=1}^K P(w_j|z_k)P(z_k|d_i) \tag{2}$$

利用贝叶斯公式, 将$$(1)$$变换为:

$$\begin{align} P(d_i,w_j) &= P(d_i)P(w_j|d_i) \\ &=P(d_i)\sum\limits_{k=1}^K P(w_j|z_k)P(z_k|d_i) \\ &= \sum\limits_{k=1}^K P(w_j|z_k)P(d_i)P(z_k|d_i) \tag{3} \\ &= \sum\limits_{k=1}^K P(z_k)P(w_j|z_k)P(d_i|z_k) \end{align}$$

使用极大似然估计, 计算PLSA模型的参数:

$$\mathcal{L}=\sum\limits_{i=1}^N\sum\limits_{j=1}^Mn(d_i, w_j)\log{P}(d_i, w_j) \tag{4}$$

其中$$n(d_i,w_j)$$是词汇$$w_j$$在文档$$d_i$$中出现的次数.

## 模型拟合

由于隐变量, 使用**EM算法**求解.

#### E步

利用当前估计参数值计算隐变量$$z$$的后验概率.

$$\begin{align} P(z_k|d_i,w_j) &= \frac{P(z_k,d_i,w_j)}{P(d_i,w_j)} \\ &= \frac{P(d_i,w_j|z_k)P(z_k)}{P(d_i)P(w_j|d_i)} \\ &= \frac{P(d_i|z_k)P(w_j|z_k)P(z_k)}{P(d_i)P(w_j|d_i)} \\ &= \frac{P(w_j|z_k)P(z_k|d_i)P(d_i)}{P(d_i)\sum\limits_{k=1}^K P(w_j|z_k)P(z_k|d_i)} \\ &= \frac{P(w_j|z_k)P(z_k|d_i)}{\sum\limits_{k=1}^K P(w_j|z_k)P(z_k|d_i)} \end{align} \tag{5}$$

#### M步

使用E步得到的$$z$$的**后验概率**, 极大化似然函数, 更新参数$$P(w_i|z_k)$$和$$P(z_k|d_i)$$.

$$\begin{align} \mathcal{L} &= \sum\limits_{i=1}^N\sum\limits_{j=1}^Mn(d_i,w_j)\log P(d_i,w_j) \\ &=  \sum\limits_{i=1}^N\sum\limits_{j=1}^Mn(d_i,w_j)\log[P(d_i)\sum\limits_{k=1}^KP(w_j|z_k)P(z_k|d_i)] \\ &= \sum\limits_{i=1}^Nn(d_i)\log P(d_i) + \sum\limits_{i=1}^N\sum\limits_{j=1}^Mn(d_i,w_j)\log\sum\limits_{k=1}^KP(w_j|z_k)P(z_k|d_i) \tag{6} \end{align}$$

其中$$n(d_i)=n(d_i,w_j)$$, 第一项是常数项, 因此极大化$$\mathcal{L}$$等价于极大化第二项:

$$\max\mathcal{L}\Leftrightarrow\max\sum\limits_{i=1}^N\sum\limits_{j=1}^Mn(d_i,w_j)\log\sum\limits_{k=1}^KP(w_j|z_k)P(z_k|d_i) \tag{7}$$

令$$\mathcal{L}_c=\sum\limits_{i=1}^N\sum\limits_{j=1}^Mn(d_i,w_j)\log\sum\limits_{k=1}^KP(w_j|z_k)P(z_k|d_i)$$, 对$$\mathcal{L}_c$$求期望得:

$$\begin{align}E(\mathcal{L}_c) &= \sum\limits_{i=1}^N\sum\limits_{j=1}^Mn(d_i,w_j)E(\log\sum\limits_{k=1}^KP(w_j|z_k)P(z_k|d_i)) \\ &= \sum\limits_{i=1}^N\sum\limits_{j=1}^Mn(d_i,w_j)\sum\limits_{k=1}^KP(z_k|d_i,w_j)\log[P(w_j|z_k)P(z_k|d_i)] \tag{8} \end{align}​$$

又有限制:

$$\sum\limits_{j=1}^M P(w_j|z_k)=1 \\ \sum\limits_{j=1}^M P(z_k|d_i)=1 \tag{9}$$

因此问题转化为**带约束条件的极大值问题**, 引入**Lagrange函数**$$\tau_{k}$$, $$\rho_{i}$$, 有:

$$\begin{align} H &= \sum\limits_{i=1}^N \sum\limits_{j=1}^M n(d_i,w_j) \sum\limits_{k=1}^K P(z_k|d_i,w_j)\log[P(w_j|z_k)P(z_k|d_i)] + \sum\limits_{k=1}^K \tau_{k}(1-\sum\limits_{j=1}^MP(w_j|z_k)) + \sum\limits_{i=1}^N \rho_{i}(1-\sum\limits_{k=1}^K P(z_k|d_i)) \\ &= \sum\limits_{i=1}^N \sum\limits_{j=1}^M n(d_i,w_j) \sum\limits_{k=1}^K P(z_k|d_i,w_j)\log(w_j|z_k)+\sum\limits_{j=1}^M n(d_i,w_j) \sum\limits_{k=1}^K P(z_k|d_i,w_j)\log(z_k|d_i)+ \\ & \sum\limits_{k=1}^K \tau_{k}(1-\sum\limits_{j=1}^MP(w_j|z_k)) + \sum\limits_{i=1}^N \rho_{i}(1-\sum\limits_{k=1}^K P(z_k|d_i)) \tag{10} \end{align}$$

因此问题转换为$$\max{H}$$. 对$$H$$求每个参数的偏导数, 并令偏导数都为0, 得到:

$$\begin{align}\frac{\partial H}{\partial P(w_j|z_k)}=\sum\limits_{i=1}^N n(d_i,w_j)P(z_k|d_i,w_j)\frac{1}{P(w_j|z_k)}-\tau_k=0, \quad j=1,\cdots,M;k=1,\cdots,K \end{align}$$

$$\begin{align}\frac{\partial H}{\partial P(z_k|d_i)}=\sum\limits_{i=1}^N n(d_i,w_j)P(z_k|d_i,w_j)\frac{1}{P(z_k|d_i)}-\rho_i=0, \quad i=1,\cdots,N;k=1,\cdots,K \end{align}$$

由$$(10)$$得到:

$$\begin{align}P(w_j|z_k)=\frac{\sum\limits_{i=1}^Nn(d_i,w_j)P(z_k|d_i,w_j)}{\tau_k}, \quad j=1,\cdots,M;k=1,\cdots,K \end{align}$$

$$\begin{align}P(z_k|d_i)=\frac{\sum\limits_{i=1}^Nn(d_i,w_j)P(z_k|d_i,w_j)}{\rho_i}, \quad i=1,\cdots,N;k=1,\cdots,K \tag{11} \end{align}$$

再利用$$(9)$$的限制, 对上面两式的两侧求和, 可得:

$$\tau_k=\sum\limits_{j=1}^M\sum\limits_{i=1}^Nn(d_i,w_j)P(z_k|d_i,w_j), k=1,\cdots,K$$

$$\rho_i=\sum\limits_{k=1}^K\sum\limits_{i=1}^Nn(d_i,w_j)P(z_k|d_i,w_j), i=1,\cdots,N \tag{12}$$

将$$(13)$$带入$$(12)$$得到最终结果:

$$\begin{align}P(w_j|z_k)=\frac{\sum\limits_{i=1}^Nn(d_i,w_j)P(z_k|d_i,w_j)}{\sum\limits_{j=1}^M\sum\limits_{i=1}^Nn(d_i,w_j)P(z_k|d_i,w_j)}, \quad j=1,\cdots,M;k=1,\cdots,K \end{align}$$

$$\begin{align}P(z_k|d_i)=\frac{\sum\limits_{i=1}^Nn(d_i,w_j)P(z_k|d_i,w_j)}{\sum\limits_{k=1}^K\sum\limits_{i=1}^Nn(d_i,w_j)P(z_k|d_i,w_j)}, \quad i=1,\cdots,N;k=1,\cdots,K \end{align}$$

E步和M步之间不断迭代, 直到收敛.

## PLSA缺点

PLSA不是一个**生成模型**. PLSA只能对训练的文本得到降维主题向量, 由$$P(z,d)$$组成. 对于新的文本, 没有方法得到主题向量.

## PLSA与LSA的联系

- 相似性

  - 将$$(3)$$重写为矩阵形式, 定义矩阵:

    $$\hat{T}=(P(d_i|z_k))_{i,k}$$

    $$\hat{S}=diag(P(z_k))_k$$

    $$\hat{D}=(P(w_j|z_k))_{j,k}$$

    则联合概率模型$$P$$可以写为$$P=\hat{T}\hat{S}\hat{D}^T$$, 与LSA的形式一致.

- 差别点
  - PLSA具有更坚实的统计学基础

