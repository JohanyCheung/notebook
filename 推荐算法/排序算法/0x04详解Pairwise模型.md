## Pairwise原理

对于用户输入的一个query $$q$$, $$U_i$$和$$U_j$$是结果中其中两篇待排序的文章, $$s_i$$是模型打给文章$$U_i$$的分数, 分数的含义是预测了$$q$$和$$U_i$$之间的相关性, $$s_j$$同理.

我们可以预测将文章$$U_i$$排在$$U_j$$之前的概率, 从而将排序转化成一个二分类问题. 将$$U_i$$排在$$U_j$$之前的概率定义如下:

$$P_{ij}=P(U_i \triangleright U_j)=\frac{1}{1+e^{-\sigma (s_i-s_j)}}$$

这里的$$\sigma$$是一个**超参数**, 并不是sigmoid函数.

用$$\bar{P}_{ij}$$定义$$U_i$$排在$$U_j$$前面这个论述的真实性, 即二分类的真实标签. 1为论述为真, 0为论述为假, 0.5代表$$U_i$$与$$U_j$$的位置相同. 做一个简单的数据变换, 用$$S_{ij}\in[-1,0,1]$$来代表$$U_i$$排在$$U_j$$前面这个论述的真实性, 从而定义损失函数**binary cross entropy loss**如下:

$$
\begin{aligned}
    C &= -\bar{P}_{ij}\log P_{ij} - (1 - \bar{P}_{ij})\log(1 - P_{ij}) \\
    &= \frac{1}{2}(1-S_{ij})\sigma(s_i-s_j) + \log(1+e^{\sigma(s_i-s_j)})
\end{aligned}
$$

其中第一步到第二步的转换是带入了$$\bar{P}_{ij}=\frac{1}{2}(1+S_{ij})$$以及$$S_{ij}\in[-1,0,1]$$.

关于训练集, 也需要进行简化. 由于一对文章$$U_i$$和$$U_j$$, $$U_i$$排在$$U_j$$前和$$U_j$$排在$$U_i$$后两个论述真实性一定是互斥的, 所以如果把$$(U_i, U_j, 1)$$$$(U_j, U_i, 0)$$都放入在训练集中, 会出现冗余的情况. 因此在训练集中, 只保留所有$$S_{ij}=1$$的样本.

然后以上面的**binary cross entropy loss**为损失函数, 对模型中所有的参数$$w_k$$求导:

$$
\begin{aligned}
    \frac{\partial C}{\partial w_k} &= \frac{\partial C}{\partial s_i}\frac{\partial s_i}{\partial w_k} + \frac{\partial C}{\partial s_j}\frac{\partial s_j}{\partial w_k} \\
    &= \sigma(\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma(s_i-s_j)}})(\frac{\partial s_i}{\partial w_k} - \frac{\partial s_j}{\partial w_k}) \\
    &= \lambda_{ij}(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})
\end{aligned}
$$

将$$(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})$$前面的项用$$\lambda_{ij}$$表示, 由于训练样本中的$$S_{ij}=1$$, 可以将$$\lambda_{ij}$$简化如下:

$$\lambda_{ij}=\sigma(\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma(s_i-s_j)}})=\frac{-\sigma}{1+e^{\sigma(s_i-s_j)}}$$

而$$\frac{\partial s_i}{\partial w_k}$$和$$\frac{\partial s_j}{\partial w_k}$$是模型的输出对参数的导数, 这个在不同的模型中有不同的解决方法(NN, GBDT等).

这里我们关心的是$$\frac{\partial C}{\partial s_i}$$和$$\frac{\partial C}{\partial s_j}$$, 且有$$\frac{\partial C}{\partial s_i}=-\frac{\partial C}{\partial s_j}$$, 因此计算得到其中的一项即可, 即**损失函数对第$$i$$篇文章的预测得分的导数**. 因为这里的导数我们用$$\lambda_{ij}$$表示, 这也是**LambdaRank**, **LambdaMART**等一系列算法的由来.

总结两点:

- $$\lambda_{ij}$$是一个梯度
- **binary loss function**, 对`列表头部的排序正确性`和`列表尾部的排序正确性`一视同仁, 实际上是优化AUC

## 优化Pairwise

而如果我们要优化**NDCG**这样重点关注头部位置的指标, 这些指标对单个文档的预测得分的微小变化, 要么无动于衷(预测得分没有引起文档排序变化), 要么反应剧烈(引起了排序变化), 因此在训练的过程中, 无法定义一个**连续可导的损失函数**.

LambdaRank/LambdaMART的解决思路, 即为既然无法定义一个符合要求的损失函数, 就不再定义损失函数了, **直接定义一个符合我们要求的梯度**. 这个不是由损失函数推导出来的, 被人为定义出来的梯度, 就是**Lambda梯度**.

如果我们现在的目标是优化NDCG指标, 如何定义梯度?

**首先**, 在上一轮迭代结束后, 将所有的文档(同一个query下的)按照上一轮迭代得到的预测分数从大到小进行排序.

**然后**对于文本对$$(U_i, U_j)$$, 如果使用binary cross entropy loss, 则损失函数对于预测分数的预测仍为:

$$\lambda_{ij}=\sigma(\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma(s_i-s_j)}})=\frac{-\sigma}{1+e^{\sigma(s_i-s_j)}}$$

**然后**, 将$$U_i$$和$$U_j$$的排序位置调换, 计算NDCG指标的变化$$|\Delta_{NDCG}|$$, 然后构造**Lambda梯度**, 将$$|\Delta_{NDCG}|$$乘到上一步得到的Lambda梯度之上, 就得到了**优化的Lambda梯度**. 不同的优化指标, 对应着不同的优化的Lambda梯度.

## 注意

- 需要特别注意样本的组织方法. 排序只对**同一个query下的候选文档/同一个推荐session下的候选商品**才有意义, 所以与普通二分类不同, **LTR训练集有一个分组的概念**, 同一个组内的候选物料才能匹配成对, 相互比较. 这一点在**xgboost**或**lightgbm**模型框架中, 对应的数据组成形式, 都有体现.

## 参考资料

- [走马观花Google TF-Ranking的源代码](https://zhuanlan.zhihu.com/p/52447211)
