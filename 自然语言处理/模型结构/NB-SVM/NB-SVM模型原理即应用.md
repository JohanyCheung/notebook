## 引入

**NBSVM**(**Naive Bayes - Support Vector Machine**)模型, 作为两个常用模型的结合, 经常用来做**文本分类**任务的baseline, 其效果往往优于两者的单独预测的效果.

单独的看NB和SVM两个模型, 都经常用来做文本分类任务的baseline, 但他们的表现非常大的程度上取决于:

- 使用的特征
- 数据集
- 模型的各种变体

论文作者总结出下列的经验:

- 对于**情感分类**问题, 使用**二值特征**(特征是否在文本中出现过)一般会取得较好的成绩
- 对于**小段文本**的**情感分类**问题, NB经常表现的比SVM好, 而对于**长文本**的情感分类问题, SVM则常常比NB表现的好
- 使用经典的SVM模型, 但如果使用的特征为来自于NB的**log-count**比率, 表现的会更好

因此, 上面的内容就是NBSVM模型的本质, 使用SVM模型进行分类, 但特征是结合NB得到的.

## 算法细节

将**线性分类器**公式化表示为, 对于第$$k$$个样本:

$$y^k=\text{sign}(\mathbf{w}^T \mathbf{x}^k + b)$$

假设这里的$$\mathbf{x}^k$$即我们的特征向量$$\mathbf{f}^i \in \mathbb{R}^{|V|}$$, $$V$$是特征集合. 且$$y^i \in \{-1, 1\}$$. 这里的特征可以是:

- 频次向量
- TF-IDF向量
- 二值向量

记:

$$\mathbf{p} = \alpha + \sum\limits_{i:y^i=1}\mathbf{f}^i$$

$$\mathbf{q} = \alpha + \sum\limits_{i:y^i=-1}\mathbf{f}^i$$

我们将上文中提到的**log-count ratio**记为:

$$\mathbf{r} = \log(\frac{\mathbf{p}/||\mathbf{p}||_1}{\mathbf{q}/||\mathbf{q}||_1})$$

如果将**MNB**模型对应成上面的线性分类器的形式, 可以从MNB的分类公式入手:

$$
\begin{aligned}
P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y) \\
\Downarrow \\
\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y)
\end{aligned}
$$

由于这里的$$y^i \in \{-1, 1\}$$, 只有两种取值方法, 因此我们可以将使用$$\arg\max$$预测分类标签, 修改为正样本对负样本统计量进行比例的方法, 再使用$$\log$$转换, 这样得到的值如果大于0则说明更偏向正样本, 小于0则偏向负样本, 因此就可以使用$$\text{sign}$$函数进行分类了:

$$
\begin{aligned}
\hat{y} &= \text{sign}(\log(\frac{P(y=1)\prod\limits_{i=1}^{n} P(x_i \mid y=1)}{P(y=0)\prod\limits_{i=1}^{n} P(x_i \mid y=0)})) \\
&= \text{sign}(\log(\frac{\prod\limits_{i=1}^{n} P(x_i \mid y=1)}{\prod\limits_{i=1}^{n} P(x_i \mid y=0)}) + \log(\frac{P(y=1)}{P(y=0)}))
\end{aligned}
$$

这样就向**线性分类器**的形式靠近了一步, 现在要做的就是确定$$\mathbf{w}^T$$, $$\mathbf{x}$$, $$b$$.

我们可以将上式中前面的一项看作是$$\mathbf{w}^T\mathbf{x}$$的结果. 如果只看第一项的分子, 是$$y=1$$的条件下, $$\mathbf{x}$$向量取对应样本值的概率, 因此是关于$$\mathbf{x}$$的一个函数, 可以表示成$$\mathbf{w}^T\mathbf{x}$$的形式. 在论文中, 取$$\mathbf{w}=\mathbf{r}$$, $$\mathbf{x}=\mathbf{f}$$, 对应的$$b=\log(N_+/N_-)$$(这一项是很直观的). 从上面可以看出$$\mathbf{r}$$的分子统计了所有$$y=1$$的向量表现, 分母统计了所有$$y=0$$的向量表现, 而且整体形式是符合上式的. 因此就得到了**MNB模型的线性分类器形式**的表现.

---

以上是MNB模型的表示方法的推理. 对于SVM模型, 根据其定义, 对于一个样本分类的推理是完全符合上面的线性分类器的表示方法的.

---

**SVM with NB features(NBSVM)**

使用MNB模型转换得到的新特征作为SVM模型训练的新特征, 具体来说, 原特征向量为$$\mathbf{\hat{f}}$$, 则经过MNB模型得到的新特征为$$\mathbf{\tilde{f}}=\mathbf{\hat{r}} \circ \mathbf{\hat{f}}$$. 通过这种方法训练得到的模型, 在**长文本**情况下表现的很好.

作者提出, **MNB模型**和**SVM模型**之间的**线性插值**的结果对于**所有形式的文本**都表现的更好, 插值的具体方式为:

$$\mathbf{w}^{'} = (1-\beta)\bar{w} + \beta \mathbf{w}$$

其中$$\bar{w}=||\mathbf{w}||_1 / |V|$$是对$$\mathbf{w}$$量级的一个均值评估. $$\beta \in [0, 1]$$. 如果$$\beta=0$$, 相当于是直接使用$$\mathbf{\tilde{f}}=\mathbf{\hat{r}} \circ \mathbf{\hat{f}}$$来做分类, 即相当于只使用**MNB**模型. 因此上式是MNB模型和SVM模型之间的一种插值.

## 变种

NBSVM模型更像是一种框架思路, 其在很多点上都有多种变种.

**特征向量**

标准的NBSVM使用的特征向量是词频向量. 论文中, 作者也提到使用二值化的向量, 即某词是否在该样本中出现过. 另外在工程实践中, 也经常使用TF-IDF向量作为特征向量.

**模型选择**

使用MNB对原始特征转换得到新特征之后, 可以考虑不使用SVM模型. 对于文本任务, 特征矩阵往往非常系数, 考虑使用**逻辑回归模型**是一个不错的选择. 在`kaggle`的一个文本分类比赛中, 其中的一个[baseline](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/notebook)就在转换之后, 使用了逻辑回归代替了SVM模型来训练, 得到了相当不错的结果.

## 参考资料

- [Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf)
- [NB-SVM strong linear baseline](https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/notebook)
- [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
