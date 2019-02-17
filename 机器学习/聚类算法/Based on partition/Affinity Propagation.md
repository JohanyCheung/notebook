## Affinity Propagation

AP(Affinity Propagation)算法, 使用一些最能代表其他样本点的**代表点**(exemplars)完成聚类, 每个代表点, 以及这个代表点关联着的所有样本点, 就是一个簇.

代表点与样本点组成的点对, 我们考虑之间的一种可能性, 即这个点可以作为代表点, 代表另一个点的可能性. 这个值在每次迭代中得到更新, 直到收敛, 从而聚类结果得到了确定.

AP算法**无须提前指定簇的的数量**, 而是根据数据集自主确定. 有两个重要的参数会影响到代表点的选择和数量, 进而影响最后聚类的表现, 分别是:

- **preference**: 点与其他多点之间的可能性(上述)大于多少时, 更会被选为代表点, 从而直接影响最后代表点的数量. 如果没有设定, 会选所有输入点之间**相似性**的中位数作为参数
- **damping**: 在上述的可能性的迭代更新时, 会使用到上一轮的这个可能性, damping将决定上轮可能性在新的可能性中的比重(衰减比例, 公式中会有直接的体现), 以避免值的震荡

AP算法的主要**缺点**是其复杂度太高了, 达到了$$O(N^2T)$$, 在稀疏的情况下有所减轻. 因此AP算法只适合使用在小样本集上.

## 算法描述

两个样本之间的关系有两个衡量公式:

- $$r(i,k)$$表示样本$$k$$是样本$$i$$的代表点(exemplar)的accumulated evidence
- $$a(i,k)$$表示样本$$i$$选择样本$$k$$作为代表点的accumulated evidence, 并且考虑了样本$$k$$作为其他所有样本的代表点的衡量值

因此, 对每个样本点进行判断, 从中选出代表点, 依据以下两个规则:

- 与足够多的样本相似
- 被很多的样本点选为代表点

而表示样本$$k$$是样本$$i$$的代表点的强度由$$r(i,k)$$表示:

$$r(i,k)=s(i,k)-\max[a(i,k^{'})+s(i,k^{'}),\ \forall k^{'}\ne k]$$

其中$$s(i,k)$$表示$$i$$和$$k$$两个样本点之间的相似度度量. 而表示样本$$i$$选择样本$$k$$作为代表点强度的度量为$$a(i,k)$$:

$$a(i,k)=\min[0,r(k,k)+\sum\limits_{i^{'},s.t.\ i^{'}\notin\{i,k\}}r(i^{'},k)]$$

在训练的开始, 将所有点对的$$r(i,k)$$和$$a(i,k)$$初始化为0, 在迭代的过程中, 为了避免值的震荡, 引入**damping factor**$$\lambda$$, 如下迭代:

$$r_{t+1}(i,k)=\lambda \cdot r_{t}(i,k) + (1 - \lambda)r_{t+1}(i,k)$$
$$a_{t+1}(i,k)=\lambda \cdot a_{t}(i,k) + (1 - \lambda)a_{t+1}(i,k)$$

## 应用

**sklearn.cluster.AffinityPropagation**对应着AP算法的模型. 初始化模型时, 无需指定簇的数量. 需要注意以下的参数:

- **damping**: float, optional, default: 0.5
  - 衰减参数, 避免更新过程中值的震荡, 具体作用即原理在上文中有体现
  - 值应当在0.5到1之间
- **max_iter**: int, optional, default: 200
  - 最大迭代次数
- **convergence_iter**: int, optional, default: 15
  - 提前停止, 如果在`convergence_iter`轮数内, 簇的数量没有变化, 则停止
- **preference**: array-like, shape (n_samples,) or float, optional
  - 对于一个(点, 其他点)对, 对应的preferences值如果超过这个值则更有可能成为代表点
  - 具体的作用见上面的阐述

更详细的内容参考`sklearn`文档:

- [Affinity Propagation](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation)
- [sklearn.cluster.AffinityPropagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation)
