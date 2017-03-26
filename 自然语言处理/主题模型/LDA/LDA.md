## 参考

[LDA|火光摇曳](http://www.flickering.cn/tag/lda/)

## 与PLSA模型的关系

PLSA模型中, **doc-topic骰子**$$\overrightarrow{\theta}_m$$和**topic-word骰子**$$\overrightarrow{\varphi}_k$$都是模型中的参数, 是确定值, 这在**贝叶斯**学派的角度看来是有问题的, 参数不应当是固定值, 而是**随机变量**, 因此就需要有**先验分布**.

## LDA模型

#### 文档生成过程

将PLSA模型对应的文档生成过程, 改造成贝叶斯的过程.

由于参数$$\overrightarrow{\theta}_m$$和$$\overrightarrow{\varphi}_k$$, 在过程中被用来多次生成一些变量, 因此观测对应于**多项分布**, 从而$$\overrightarrow{\theta}_m$$和$$\overrightarrow{\varphi}_k$$的**先验分布**好的选择就是**Drichlet分布**. 这种选择是依据了Drichlet分布和多项分布的**共轭性**. 这样我们就得到了**LDA(Latent Dirichlet Allocation)**模型.

假设文档是按照下面的规则生成的:

![](http://www.52nlp.cn/wp-content/uploads/2013/02/game-lda-1.jpg)

#### 数学表示

用数学符号与公式来表示:

假设语料库中有$$M$$篇文档, 所有文档中的单词(word)和单词对应的主题(topic)如下表示:

$$\begin{align*}  \overrightarrow{\mathbf{w}} & = (\overrightarrow{w}_1, \cdots, \overrightarrow{w}_M) \\  \overrightarrow{\mathbf{z}} & = (\overrightarrow{z}_1, \cdots, \overrightarrow{z}_M)  \end{align*}$$

其中, $$\overrightarrow{w}_m$$表示第$$m$$篇文档中的词, 因此相互之间的向量长度并不相等; $$\overrightarrow{z}_m$$表示这些词对应的topic编号, 长度同$$\overrightarrow{w}_m$$相等, 是一一对应的关系. 如下图所示:

![](http://www.52nlp.cn/wp-content/uploads/2013/02/word-topic-vector.jpg)

#### 物理过程分解

每个文档中的每个词生成过程如下:

- $$\overrightarrow{\alpha}\rightarrow \overrightarrow{\theta}_m \rightarrow z_{m,n}$$
  - $$\overrightarrow{\alpha}\rightarrow \overrightarrow{\theta}_m$$: 对于第$$m$$篇文档, 随机抽取了一个**doc-topic**骰子$$\overrightarrow{\theta}_m$$, 拥有$$K$$个面
  - $$\overrightarrow{\theta}_m \rightarrow z_{m,n}$$: 投掷上述的骰子, 生成了文档$$m$$中的第$$n$$个词的主题$$z_{m,n}$$
- $$\overrightarrow{\beta} \rightarrow \overrightarrow{\varphi}_k \rightarrow w_{m,n} | k=z_{m,n}$$
  - $$\overrightarrow{\beta} \rightarrow \overrightarrow{\varphi}_k$$: 对于$$K$$个骰子, 挑选topic为$$k=z_{m,n}$$对应的那个骰子进行投掷
  - $$\overrightarrow{\varphi}_k \rightarrow w_{m,n}$$: 使用挑选出来的骰子$$\overrightarrow{\varphi}_k$$进行投掷, 得到单词$$w_{m,n}$$

因此整个LDA被分解成两个**物理过程**. 由此, LDA生成模型中, $$M$$篇文档对应于$$M$$个独立的**Dirichlet-Multinomial共轭结构**, $$K$$个topic对应于$$K$$个独立的**Dirichlet-Multinomial共轭结构**. 因此LDA模型对应着$$M+K$$个**Dirichlet-Multinomial共轭结构**.

理解LDA的关键就在于它是如何被分解成$$M+K$$个共轭结构的.

#### Dirichlet-Multinomial共轭结构

- $$\overrightarrow{\alpha}\rightarrow \overrightarrow{\theta}_m \rightarrow \overrightarrow{z}_{m}$$

  表示生成第$$m$$篇文档中所有词对应的topics. $$\overrightarrow{\alpha}\rightarrow \overrightarrow{\theta}_m$$对应于 Dirichlet 分布, $$\overrightarrow{\theta}_m \rightarrow \overrightarrow{z}_{m}$$对应于 Multinomial 分布, 所以整体是一个Dirichlet-Multinomial 共轭结构.

  借助Dirichlet-Multinomial共轭结构的性质:

  $$\begin{align}  p(\mathcal{W}|\overrightarrow{\alpha}) & = \int p(\mathcal{W}|\overrightarrow{p}) p(\overrightarrow{p}|\overrightarrow{\alpha})d\overrightarrow{p} \notag \\  & = \int \prod_{k=1}^V p_k^{n_k} Dir(\overrightarrow{p}|\overrightarrow{\alpha}) d\overrightarrow{p} \notag \\  & = \int \prod_{k=1}^V p_k^{n_k} \frac{1}{\Delta(\overrightarrow{\alpha})}  \prod_{k=1}^V p_k^{\alpha_k -1} d\overrightarrow{p} \notag \\  & = \frac{1}{\Delta(\overrightarrow{\alpha})}  \int \prod_{k=1}^V p_k^{n_k + \alpha_k -1} d\overrightarrow{p} \notag \\  & = \frac{\Delta(\overrightarrow{n}+\overrightarrow{\alpha})}{\Delta(\overrightarrow{\alpha})}  \label{likelihood-dir-mult}  \end{align}$$

  可以得到:

  $$p(\overrightarrow{z}_m |\overrightarrow{\alpha}) = \frac{\Delta(\overrightarrow{n}_m+\overrightarrow{\alpha})}{\Delta(\overrightarrow{\alpha})}$$

  其中, $$\overrightarrow{n}_m = (n_{m}^{(1)}, \cdots, n_{m}^{(K)})$$, $$n_{m}^{(k)}$$表示第$$m$$篇文档中第$$k$$个topic产生的词的数量.

  而且$$\overrightarrow{\theta}_m$$的后验分布为:

  $$Dir(\overrightarrow{\theta}_m| \overrightarrow{n}_m + \overrightarrow{\alpha})$$

  而$$M$$篇文档相互独立, 因此得到$$M$$个相互独立的Dirichlet-Multinomial共轭结构. 从而整个语料的topics生成概率为:

  $$\begin{align}  \label{corpus-topic-prob}  p(\overrightarrow{\mathbf{z}} |\overrightarrow{\alpha}) & = \prod_{m=1}^M p(\overrightarrow{z}_m |\overrightarrow{\alpha}) \notag \\  &= \prod_{m=1}^M \frac{\Delta(\overrightarrow{n}_m+\overrightarrow{\alpha})}{\Delta(\overrightarrow{\alpha})} \quad\quad  (*)  \end{align}$$

---

生成过程也可以改成, 不区分文档, 先对所有文档中的所有单词生成对应的topic, 再根据每个词给定的topic生成单词.

因此, 把语料中的词进行交换, 把具有相同topic的词放在一起, 组成$$K$$个向量:

$$\begin{align*}  \overrightarrow{\mathbf{w}}’ &= (\overrightarrow{w}_{(1)}, \cdots, \overrightarrow{w}_{(K)}) \\  \overrightarrow{\mathbf{z}}’ &= (\overrightarrow{z}_{(1)}, \cdots, \overrightarrow{z}_{(K)})  \end{align*}$$

$$\overrightarrow{w}_{(k)}$$表示由第$$k$$个topic生成的词, 长度为第$$k$$个topic生成的词的数量. $$\overrightarrow{z}_{(k)}$$表示这些词对应的topic, 因此在$$\overrightarrow{z}_{(k)}$$向量中, 所有的元素都是$$k$$.

- $$\overrightarrow{\beta} \rightarrow \overrightarrow{\varphi}_k \rightarrow w_{m,n} | k=z_{m,n}$$

  $$\overrightarrow{\beta} \rightarrow \overrightarrow{\varphi}_k$$对应于 Dirichlet 分布, $$\overrightarrow{\varphi}_k \rightarrow \overrightarrow{w}_{(k)}$$对应于 Multinomial 分布, 整体也还是一个 Dirichlet-Multinomial 共轭结构.

  同理得到:

  $$p(\overrightarrow{w}_{(k)} |\overrightarrow{\beta}) = \frac{\Delta(\overrightarrow{n}_k+\overrightarrow{\beta})}{\Delta(\overrightarrow{\beta})}$$

  其中$$\overrightarrow{n}_k = (n_{k}^{(1)}, \cdots, n_{k}^{(V)})$$, $$n_{k}^{(t)}$$表示第$$k$$个topic产生的词中单词$$t$$的个数.

  而且$$\overrightarrow{\varphi}_k$$的后验分布为:

  $$Dir( \overrightarrow{\varphi}_k| \overrightarrow{n}_k + \overrightarrow{\beta}).$$

  $$K$$个topics生成单词的过程相互独立, 所以$$K$$个Dirichlet-Multinomial 共轭结构相互独立, 则整个语料中词生成的概率为:

  $$\begin{align}  \label{corpus-word-prob}  p(\overrightarrow{\mathbf{w}} |\overrightarrow{\mathbf{z}},\overrightarrow{\beta}) &= p(\overrightarrow{\mathbf{w}}’ |\overrightarrow{\mathbf{z}}’,\overrightarrow{\beta}) \notag \\  &= \prod_{k=1}^K p(\overrightarrow{w}_{(k)} | \overrightarrow{z}_{(k)}, \overrightarrow{\beta}) \notag \\  &= \prod_{k=1}^K \frac{\Delta(\overrightarrow{n}_k+\overrightarrow{\beta})}{\Delta(\overrightarrow{\beta})}  \quad\quad (**)  \end{align}$$

  结合上面两式, 得到:

  $$\begin{align}  \label{lda-corpus-likelihood}  p(\overrightarrow{\mathbf{w}},\overrightarrow{\mathbf{z}} |\overrightarrow{\alpha}, \overrightarrow{\beta}) &=  p(\overrightarrow{\mathbf{w}} |\overrightarrow{\mathbf{z}}, \overrightarrow{\beta}) p(\overrightarrow{\mathbf{z}} |\overrightarrow{\alpha}) \notag \\  &= \prod_{k=1}^K \frac{\Delta(\overrightarrow{n}_k+\overrightarrow{\beta})}{\Delta(\overrightarrow{\beta})}  \prod_{m=1}^M \frac{\Delta(\overrightarrow{n}_m+\overrightarrow{\alpha})}{\Delta(\overrightarrow{\alpha})}  \quad\quad (***)  \end{align}$$

  得到了**联合分布**$$p(\overrightarrow{\mathbf{w}},\overrightarrow{\mathbf{z}})$$.

#### Gibbs Sampling

有了联合分布$$p(\overrightarrow{\mathbf{w}},\overrightarrow{\mathbf{z}})$$, 就可以使用**MCMC算法**进行采样了, 这里使用**Gibbs Sampling**进行采样.

由于$$\overrightarrow{\mathbf{w}}$$是观测值, 是已知的, 只有$$\overrightarrow{\mathbf{z}}$$是隐含变量, 因此真正需要采样的是$$p(\overrightarrow{\mathbf{z}}|\overrightarrow{\mathbf{w}})$$.

对于语料库中的第$$i$$个词(这里有$$i=(m,n)$$, $$i$$是一个二维坐标, 对应第$$m$$篇文档中的第$$n$$个词), 其在$$\overrightarrow{\mathbf{z}}$$对应的topic为$$z_i$$, 用$$\neg i$$表示去除下标为$$i$$这个词. 需要得到在任一坐标轴$$i$$对应的条件分布$$p(z_i = k|\overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}})$$. 假设观测到词$$w_i = t$$, 根据贝叶斯法则, 得到:

$$\begin{align*}  p(z_i = k|\overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}) \propto  p(z_i = k, w_i = t |\overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}_{\neg i}) \\  \end{align*}$$

而$$z_i = k, w_i = t$$只会涉及到第$$m$$篇文档的第$$k$$个topic, 也只会涉及到两个Dirichlet-Multinomial 共轭结构:

- $$\overrightarrow{\alpha} \rightarrow \overrightarrow{\theta}_m \rightarrow \overrightarrow{z}_{m}$$
- $$\overrightarrow{\beta} \rightarrow \overrightarrow{\varphi}_k \rightarrow \overrightarrow{w}_{(k)}$$

由于$$M+K$$个Dirichlet-Multinomial 共轭结构相互独立, 因此有$$\overrightarrow{\theta}_m, \overrightarrow{\varphi}_k$$的后验分布为:

$$\begin{align*}  p(\overrightarrow{\theta}_m|\overrightarrow{\mathbf{z}}_{\neg i},\overrightarrow{\mathbf{w}}_{\neg i})  &= Dir(\overrightarrow{\theta}_m| \overrightarrow{n}_{m\neg i} + \overrightarrow{\alpha}) \\  p(\overrightarrow{\varphi}_k|\overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}_{\neg i})  &= Dir( \overrightarrow{\varphi}_k| \overrightarrow{n}_{k\neg i} + \overrightarrow{\beta})  \end{align*}$$

使用上面两个式子, 就得到了如下的Gibbs Sampling公式的推导:

$$\begin{align*}  p(z_i = k|\overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}) & \propto  p(z_i = k, w_i = t |\overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}_{\neg i}) \\  &= \int p(z_i = k, w_i = t, \overrightarrow{\theta}_m,\overrightarrow{\varphi}_k |  \overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}_{\neg i}) d \overrightarrow{\theta}_m d \overrightarrow{\varphi}_k \\  &= \int p(z_i = k, \overrightarrow{\theta}_m|\overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}_{\neg i})  \cdot p(w_i = t, \overrightarrow{\varphi}_k | \overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}_{\neg i})  d \overrightarrow{\theta}_m d \overrightarrow{\varphi}_k \\  &= \int p(z_i = k |\overrightarrow{\theta}_m) p(\overrightarrow{\theta}_m|\overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}_{\neg i})  \cdot p(w_i = t |\overrightarrow{\varphi}_k) p(\overrightarrow{\varphi}_k|\overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}_{\neg i})  d \overrightarrow{\theta}_m d \overrightarrow{\varphi}_k \\  &= \int p(z_i = k |\overrightarrow{\theta}_m) Dir(\overrightarrow{\theta}_m| \overrightarrow{n}_{m,\neg i} + \overrightarrow{\alpha}) d \overrightarrow{\theta}_m \\  & \hspace{0.2cm} \cdot \int p(w_i = t |\overrightarrow{\varphi}_k) Dir( \overrightarrow{\varphi}_k| \overrightarrow{n}_{k,\neg i} + \overrightarrow{\beta}) d \overrightarrow{\varphi}_k \\  &= \int \theta_{mk} Dir(\overrightarrow{\theta}_m| \overrightarrow{n}_{m,\neg i} + \overrightarrow{\alpha}) d \overrightarrow{\theta}_m  \cdot \int \varphi_{kt} Dir( \overrightarrow{\varphi}_k| \overrightarrow{n}_{k,\neg i} + \overrightarrow{\beta}) d \overrightarrow{\varphi}_k \\  &= E(\theta_{mk}) \cdot E(\varphi_{kt}) \\  &= \hat{\theta}_{mk} \cdot \hat{\varphi}_{kt} \\  \label{gibbs-sampling-deduction}  \end{align*}$$

推导过程中的概率物理意义是简单明了的: $$z_i = k, w_i = t$$的概率只和两个Dirichlet-Multinomial共轭结构关联. 最终得到的$$\hat{\theta}_{mk}, \hat{\varphi}_{kt}$$就是对应的两个Dirichlet后验分布在贝叶斯框架下的参数估计, 借助于Dirichlet分布的参数估计的公式, 得到:

$$\begin{align*}  \hat{\theta}_{mk} &= \frac{n_{m,\neg i}^{(k)} + \alpha_k}{\sum_{k=1}^K (n_{m,\neg i}^{(k)} + \alpha_k)} \\  \hat{\varphi}_{kt} &= \frac{n_{k,\neg i}^{(t)} + \beta_t}{\sum_{t=1}^V (n_{k,\neg i}^{(t)} + \beta_t)}  \end{align*}$$

最终得到了 LDA 模型的 Gibbs Sampling 公式:

$$\begin{equation}  \label{gibbs-sampling}  p(z_i = k|\overrightarrow{\mathbf{z}}_{\neg i}, \overrightarrow{\mathbf{w}}) \propto  \frac{n_{m,\neg i}^{(k)} + \alpha_k}{\sum_{k=1}^K (n_{m,\neg i}^{(k)} + \alpha_k)}  \cdot \frac{n_{k,\neg i}^{(t)} + \beta_t}{\sum_{t=1}^V (n_{k,\neg i}^{(t)} + \beta_t)}  \end{equation}$$

#### Training and Inference

有了 LDA 模型, 目标有两个:

- 估计模型中的参数$$\overrightarrow{\varphi}_1, \cdots, \overrightarrow{\varphi}_K$$和$$\overrightarrow{\theta}_1, \cdots, \overrightarrow{\theta}_M$$
- 对于新来的一篇文档, 能够计算这篇文档的topic分布$$\overrightarrow{\theta}_{new}$$

有了Gibbs Sampling 公式, 就可以基于语料**训练**LDA模型. 训练的过程就是获取语料中的$$(z,w)$$样本, 而模型中的所有的参数都可以基于最终采样得到的样本进行估计, 训练流程如下:

![](http://www.52nlp.cn/wp-content/uploads/2013/02/lda-training.jpg)

由这个**topic-word频率矩阵**, 我们可以计算每一个$$p(word|topic)$$概率, 从而算出模型参数$$\overrightarrow{\varphi}_1, \cdots, \overrightarrow{\varphi}_K$$, 这就是$$K$$个**topic-word骰子**.

对于新来的文档, 只要认为 Gibbs Sampling 公式中的$$\hat{\varphi}_{kt}$$, 是由训练语料得到的模型提供的, 所以采样过程中我们只要估计该文档的topic分布$$\overrightarrow{\theta}_{new}$$就好了, 预测的过程如下:

![](http://www.52nlp.cn/wp-content/uploads/2013/02/lda-inference.jpg)

