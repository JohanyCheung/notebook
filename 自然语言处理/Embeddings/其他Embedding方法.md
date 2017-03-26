其他的Embedding方法, 或者基于word2vec和GloVe方法优化的思路有:

- [Linguistic Regularities in Sparse and Explicit Word Representations](https://www.cs.bgu.ac.il/~yoavg/publications/conll2014analogies.pdf): 借助**positive pointwise mutual information, PPMI**矩阵得到词向量, 具体方法见论文. 这是一种**稀疏**的表示方法
- [Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics](http://aclweb.org/anthology/D17-1023): 从方法名称就可以看出, 用**n-gram**代替了单个单词**word**. 原本的word2vec可以看成1gram2vec. 由于使用了n-gram从语料中进行提取, 因此*词*(这里的词指的是一个n-gram中所有单词组成的整体)的数量会爆炸, 论文中表述了相应的应对方法. 这种算法是借助于word2vec, GloVe, PPMI等基础算法, 进行的优化.
