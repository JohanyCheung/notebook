## 引入

朴素贝叶斯方法是一系列的监督学习方法, 其中**朴素**(naive)的含义是假设**特征之间是相互独立的**. 给定类别$$y$$和特征向量$$\mathbf{x}$$之后, 有:

$$P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots x_n \mid y)}{P(x_1, \dots, x_n)}$$

根据**朴素性**, 即特征之间相互独立的性质, 有:

$$P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y),$$

因此, 朴素贝叶斯通过下面的方法进行分类:

$$
\begin{aligned}
P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y) \\
\Downarrow \\
\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y)
\end{aligned}
$$

可以使用**最大后验概率**(Maximum A Posteriori, MAP)来对$$P(y)$$和$$P(x_i \mid y)$$进行估计.

**各种朴素贝叶斯分类器之间的区别, 主要在于假设的$$P(x_i \mid y)$$的分布形式的不同**.

尽管假设条件过于理想, 朴素贝叶斯分类器在显示中很多情况下的表现还是很不错的, 例如:

- 文档分类
- 垃圾邮件分类

等各种分类问题. 仅需要**很少量**的数据来训练模型的参数. 关于朴素贝叶斯表现良好的理论解释, 在[The optimality of Naive Bayes](http://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf)中有详细的说明.

此外, 相对于其他模型, 朴素贝叶斯一个巨大的优点是**训练速度超级快**.

但尽管朴素贝叶斯模型是一个表现相当不错的分类器, 但又是一个相当糟糕的`estimator`, 这里指的是它预测得到的每个类别的概率(`predict_proba`)是很不可靠的, 没有多少参考价值.

## 多种朴素贝叶斯模型

### Gaussian Naive Bayes

对应于[sklearn.naive_bayes.GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB)模型. 模型中, 关于特征的似然函数被假设为服从如下的**高斯分布**:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)$$

训练过程就是使用最大似然法估计分布参数$$\sigma_y$$和$$\mu_y$$.

### Multinomial Naive Bayes

对应于[sklearn.naive_bayes.MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)模型. 适用于符合**多项分布**的数据. 多使用在**text classification**场景中, 例如特征是**word counts**或者**tf-idf vectors**. 假设有$$n$$个特征, 对于每一个类$$y$$, 特征向量的分布服从如下的参数向量$$\theta_y = (\theta_{y1},\ldots,\theta_{yn})$$. 其中$$\theta_{yi}$$即是$$P(x_i \mid y)$$, 即对于一个样本, 特征$$i$$在类别$$y$$出现的概率.

训练的过程即是评估参数$$\theta_y$$的过程, 这里使用带有平滑的最大似然估计, 即参数由以下的方式得到:

$$\hat{\theta}_{yi} = \frac{ N_{yi} + \alpha}{N_y + \alpha n}$$

其中$$N_{yi} = \sum_{x \in T} x_i$$是训练集中, 特征$$i$$在类别$$y$$中出现的次数, $$N_{y} = \sum_{i=1}^{n} N_{yi}$$是类别$$y$$中所有特征出现的总次数.

加入平滑项$$\alpha \ge 0$$防止在预测中, 某个特征永远不会出现的情况. 其中, 当$$\alpha = 1$$时, 称为**Laplace smoothing**, 当$$\alpha \lt 1$$时称为**Lidstone smoothing**.

### Complement Naive Bayes

[sklearn.naive_bayes.ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB)是标准的multinomial naive Bayes(MNB)的一种变形. 特别适用于**不平衡的数据集**. CNB使用每个类别统计量的互补部分来计算模型的参数权重. 相对于MNB来说, CNB估计得到的参数更稳定, 因此在text classification任务中往往有由于MNB的表现. 模型的参数如下得到:

$$
\begin{aligned}
\hat{\theta}_{ci} = \frac{\alpha_i + \sum_{j:y_j \neq c} d_{ij}}{\alpha + \sum_{j:y_j \neq c} \sum_{k} d_{kj}} \\
w_{ci} = \log \hat{\theta}_{ci} \\
w_{ci} = \frac{w_{ci}}{\sum_{j} |w_{cj}|}
\end{aligned}
$$

可以看到, 对于类别$$c$$相关参数的计算, 是在所有非$$c$$类的样本中加和计算得到的. 其中$$d_{ij}$$是样本$$j$$中特征项$$i$$的值, 可以是出现的次数统计或者`tf-idf`的值. 而且此时的平滑项跟细致, 对于每个特征$$i$$都有对应的平滑项$$\alpha_i$$.

此外, 为了消除长样本对于模型参数的较大影响, 使用如下的方法预测样本的分类结果:

$$\hat{c} = \arg\min_c \sum_{i} t_i w_{ci}$$

样本被指认为**补足量**最小的类, 且与每个特征在该样本中的出现次数$$t_i$$相关.

相关论文见: [Tackling the poor assumptions of naive bayes text classifiers](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf).

### Bernoulli Naive Bayes

[sklearn.naive_bayes.BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB)假设所有特征都是符合**伯努利分布**的, 即特征值是**二值的**. 如果使用在文本分类任务中, 对应的特征就是该单词是否在这个样本中出现过. 程序中如果传入的训练数据是非二值的, 模型也会自动地将其转换成二值类型.

由于$$x_i$$是二值的, 因此有:

$$P(x_i \mid y) = P(i \mid y) x_i + (1 - P(i \mid y)) (1 - x_i)$$

由于它的训练过程与MNB类似, 只是特征值的不同, 因此如果时间允许, 最好使用两套特征分别训练MNB和BNB. 特别的, BNB适用于样本较短的情况(评论, 微博等情况).

## 参考资料

- [Naive Bayes - sklearn document](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [朴素贝叶斯 - sklearn中文文档](http://sklearn.apachecn.org/#/docs/10)
