## 嵌入

#### 定义

**嵌入**(Embedding)技术是将每个**离散变量**表示成**连续向量**的方法. 对于**词嵌入**(Word Embedding)而言, 就是将每个单词由一个数值连续的向量表示, 转换后的词向量继续输入到下游模型中进行其他的计算.

#### 方法

这里只以**词嵌入**进行说明, 其余情况下的嵌入方法可以类比.

1. **One Hot**

   **One Hot**方法是一种简单直观的**向量化**方法. 对于由$$N$$个词组成的词典, 使用One Hot方法, 会将每个词转换成一个长度为$$N$$的向量, 按照单词在词典中对应的索引, 每个单词向量只有在其对应的索引处值为1, 向量中其他位置的值全部为0.

   但这种方法的缺点显而易见:

   - 向量长度与词典大小成线性关系, 单词数量爆炸时, 向量长度过长, 且极度稀疏. 造成计算资源和内存资源的巨大浪费

   - 每个词之间是相互孤立的, 相互之间完全正交. 语意相似词汇和相反词汇之间的关系完全一样. 在向量空间中近似的单词不能处在空间中相似的位置

2. **学习嵌入**

   嵌入(Embedding)方法, 就是通过**学习**获取一个指定长度的相对低维的向量. 具体来说:

   通过构建一个**监督任务**, 使用**神经网络**来学习每个单词对应的向量, 即学习一个**嵌入矩阵**(Embedding Matrix). 这样得到的**嵌入向量**是每个词的表征, 其中的相似词在嵌入空间中的**距离**会更近, 即此时空间中点之间的距离就有了意义.

   而我们构建的这个监督任务(常常是一种分类任务), 其目的不是为了最终的预测结果, 而是为了获取**嵌入权重**, 这个模型预测本身仅仅是为了实现这个目的的一种形式. 找到如何创建监督式任务以得出相关表征的方法是嵌入设计中关键的部分.

#### 用途

有了嵌入向量, 通常有下面几个用途:

- 作为机器学习模型的**输入**学习监督任务
- 通过嵌入空间中的距离, 做**比较**(如定量这两相似性)和**查找**(如找到最相似)等工作
- **可视化**

#### 嵌入的类比推理

嵌入向量在空间中的位置是具有一定意义的, 从词语的类比推理中可以看出.

对于词语之间的映射关系$$\text{man} \to \text{woman}$$, 而对于$$\text{king} \to \text{?}$$我们知道是`queen`. 而且这四个词语的嵌入向量上符合一种有趣的关系:

$$e_{man}-e_{woman} \approx e_{king}-e_{queen}$$

即左右两个**差向量**是相似的, 即**语义上的相似性**在空间中也是相似的. 那么在实际中, 就可以通过一些相似度函数(如余弦相似度函数)来做一些比较和查找的工作.

#### 学习方法

学习嵌入权值的方法一般是通过构建一个**语言模型**来学习. 这种语言模型是**向量形式**的, **面向语义**的. 即两个语义相似的词对应的词向量也是相似的, 具体反映在向量的夹角或点的距离上.

常见的方法有:

- **word2vec**: 其本身是语言模型学习得到的**中间结果**, 有以下两种形式:

  - **CBOW**
  - **Skip-gram**

  模型为了降低复杂度又采用了**Hierarchical Softmax**和**Negative Sampling**两种方法. **Skip-gram with negative sampling(SGNS)**是一种最为常用的学习嵌入的方法.

  相关论文: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).

  对于**罕见词**表现差的问题, 提出了优化方法: [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606).

- **[GloVe](https://nlp.stanford.edu/pubs/glove.pdf)**: 一种简便的学习词嵌入的方法

- **PPMI & SVD**: 无需通过构建语义任务训练得到.

  [Linguistic regularities in sparse and explicit word representations](https://www.cs.bgu.ac.il/~yoavg/publications/conll2014analogies.pdf)

  [Neural word embedding as implicit matrix factorization](http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization)

- **Ngram**: 对于以上三种方法, 使用语义模型中常用的**词袋模型**.

  [Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics](http://aclweb.org/anthology/D17-1023)

