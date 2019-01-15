[这篇论文](https://openreview.net/pdf?id=SyK00v5xx)提出了**SIF sentence embedding**方法, 作者提供的代码在[Github](https://github.com/PrincetonML/SIF).

## 引入

作为一种**无监督**计算句子之间相似度的方法, **sif sentence embedding**使用预训练好的词向量, 使用加权平均的方法, 对句子中所有词对应的词向量进行计算, 得到整个句子的embedding向量. 再使用句子向量进行相似度的计算.

在这篇论文之前, 也有与这篇文章思路非常相近的思路, 即都是使用**词向量**, 通过平均的方法得到**句子向量**, 只是在加权时**权重计算方法**上有区别. 具体来说有:

- 对句子中所有单词直接求平均, 每个单词的权重相同, 得到sentence embedding
- 使用每个词的**TF-IDF**值为权重, 加权平均, 得到sentence embedding

这篇论文使用**smooth inverse frequency, sif**作为每个单词的权重, 代替**TF-IDF**值, 获得了更好的效果. 除了使用新的词权重计算方法, 还在加权平均后, 减掉了**principal component**, 最终得到句子的embedding.

另外论文中还提到了这种方法的**鲁棒性**:

- 使用不同语料(多种领域)训练得到的不同的**word embedding**, 均取得了很好的效果, 说明了对各种语料的友好.
- 使用不同语料得到的**词频**, 作为计算词权重的因素, 对最终的结果影响很小.
- 对于方法中的**超参数**, 在很大范围内, 获得的结果都是区域一直的, 即超参数的选择没有太大的影响.

## 理论

#### 1. 生成模型

首先从**潜变量生成模型**(latent variable generative model)说起. 这个模型假设: 语料的生成是一个动态过程(dynamic process), 即第$$t$$个单词是在第$$t$$步生成的.

每个单词$$w$$对应着一个$$\mathbb{R}^d$$维的向量. 而这个动态过程是由**discourse vector**$$c_t\in{\mathbb{R}^d}$$的**随机游走**驱动的. **discourse vector**代表着这个句子*what is being talked about*, 作为**潜变量**, 代表着句子一个状态, 由于是动态的, 这个状态是随时间变化的, 因此记为$$c_t$$.

单词$$w$$的向量$$v_w$$与当前时间的**discourse vector**$$c_t$$的**内积**, 表示着这个单词与整个句子之间的关系. 并且我们假设$$t$$时刻观测到单词$$w$$的概率为这个内积的**对数线性**(log linear)关系:

$$Pr(\text{w emitted at time t}| c_t)\propto{\exp(\langle c_t,v_w \rangle)}$$

由于$$c_t$$是较小幅度的**随机游走**得到的, $$c_t$$与$$c_{t+1}$$之间只会差一个较小的随机差向量, 因此**相邻的单词**是由近似的**discourse vector**生成得到的. 另外计算表明这种模型的随机游走允许偶尔$$c_t$$有较大的jump, 这对共生概率的影响是很小的.

通过这种办法生成的单词向量, 与**word2vec(CBOW)**和**Glove**生成的向量是相似的.

#### 2. 随机游走模型的改进

借助上面的模型, 我们希望如下获得一个句子的我**sentence embedding**: 对**discourse vector**做**最大似然估计**. 为了简化, 注意到$$c_t$$在整个句子生成单词的过程中, 变化很小, 因此我们将所有步的**discourse vector**假设为一个固定的向量$$c_s$$. 可证明: 对$$c_s$$的最大似然估计就是对所有单词embedding向量的平均.

这篇论文对这种模型进行了改进, 加入了**两项平滑项**, 出于如下的考虑:

- 有些单词在规定的上下文范围之外出现, 也可能对**discourse vector**产生影响
- 有限单词的出现(如常见的停止词)与**discourse vector**没有关系

出于这两点考虑, 引入了两种平滑项, 首先是对数线性模型中的一个累加项(additive term)$$\alpha p(w)$$,其中$$p(w)$$是单词$$w$$在整个语料中出现的概率(词频角度), $$\alpha$$是一个**超参数**. 这样, 即使和$$c_s$$的内积很小, 这个单词也有概率出现.

然后, 引入一个纠正项, **common discourse vector**$$c_0\in{\mathbb{R}^d}$$, 其意义是句子的*最频繁的意义*, 可以认为是句子中**最重要的成分**, 常常可以与**语法**联系起来. 文章中认为对于某个单词, 其沿着$$c_0$$方向的成分较大(即向量投影更长), 这个纠正项就会提升这个单词出现的概率.

校正后, 对于给定的**discourse vector**$$c_s$$, 单词$$w$$在句子$$s$$中出现的概率为:

$$Pr(\text{w emitted in sentence s}| c_s)\propto{\alpha p(w) + (1-\alpha)\frac{\exp(\langle \tilde{c}_s, v_w \rangle)}{Z_{\tilde{c}_s}}}$$

其中, $$\tilde{c}_s=\beta c_0+(1-\beta)c_s,\ c_0\perp c_s$$, $$\alpha$$和$$\beta$$都是**超参数**, $$Z_{\tilde{c}_s}=\sum\limits_{w\in{V}}\exp(\langle \tilde{c}_s, v_w \rangle)$$是**归一化常数**. 从公式中可以看出, 一个与$$c_s$$没有关系的单词$$w$$, 也可以在句子中出现, 原因有:

- 来自$$\alpha p(w)$$项的数值
- 与**common discourse vector** $$c_0$$的相关性

#### 3. 计算句子向量

句子向量就是上述模型中的$$c_s$$, 使用**最大似然法**估计$$c_s$$向量. 首先假设所有单词的向量$$v_s$$是大致**均匀分布**在整个向量空间上的, 因此这里的归一化项$$Z_c$$对于不同的句子值都是大致相同的, 即对于任意的$$\tilde{c}_s$$, $$Z$$值是相同的. 在此前提下, 得到似然函数:

$$p[s|c_s]=\prod\limits_{w\in{s}}p(w|c_s)=\prod\limits_{w\in{s}}[\alpha p(w) + (1-\alpha)\frac{\exp(\langle \tilde{c}_s, v_w \rangle)}{Z}]$$

取对数, 单个单词记为

$$f_w(\tilde{c}_s)=\log[\alpha p(w) + (1-\alpha)\frac{\exp(\langle \tilde{c}_s, v_w \rangle)}{Z}]$$

最大化上式, 具体的推到在论文中有详述的说明, 最终目标为:

$$\arg\max\limits_{c_s}\sum\limits_{w\in{s}}f_w(\tilde{c}_s)$$

可以得到:

$$\tilde{c}_s\propto \sum\limits_{w\in{s}}\frac{a}{p(w)+a}v_w,\ a=\frac{1-\alpha}{\alpha Z}$$

因此可以得到:

- 最优解为句子中所有单词向量的**加权平均**
- 对于词频更高的单词$$w$$, 权值更小, 因此这种方法也等同于**下采样**频繁单词

最后, 为了得到最终的**句子向量**$$c_s$$, 我们需要估计$$c_0$$. 通过计算向量$$\tilde{c}_s$$的**first principal component**(**PCA**中的主成分), 将其作为$$c_0$$. 最终的句子向量即为$$\tilde{c}_s$$减去主成份向量$$c_0$$.

#### 4. 算法总结

整个算法步骤总结如下图:

![](https://img-blog.csdn.net/20170524145901481?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMzExODg2MjU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

