## Transformer

本文介绍了**Transformer**结构, 是一种**encoder-decoder**, 用来处理序列问题, 常用在NLP相关问题中. 与传统的专门处理序列问题的encoder-decoder相比, 有以下的特点:

- 结构完全**不依赖于CNN和RNN**
- 完全依赖于**self-attention**机制, 是一种**堆叠的self-attention**
- 使用**全连接层**
- 逐点**point-wise**计算的

整个Transformer的结构图如下所示:

![](http://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWibp1593y9ib5hyUv34YYrkDnp7EuSmMARftO8gbg2OYZJ7ECbX1UaforwIOFRJ3iavJk9m1CCbEV7zA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## Encoder and Decoder Stacks
如上所说, Transformer是基于**stacked self-attention**的, **stack**方式具体为:

#### Encoder

Encoder是由$$N=6$$个独立的层堆叠而成的, 每层有两个子层:

- 第一个层为**multi-head self-attention**结构
- 第二层为**simple, position-wise fully connected feed-forward network**, 即基于位置的简单全连接反馈网络

在每个子层又引入了**residual connection**, 具体的做法为每个子层的输入与输出**相加**, 这就要求每个子层的输入与输出的维度是完全相等的. 然后再使用**layer normalization**. 因此每个子层的最终输出为:

$$LayerNorm(x + Sublayer(x))$$

此外, 论文中限定了**Embeding层**的输出和两个子层的输入输出的维度都是$$d_{model}=512$$.

#### Decoder

Decoder也是由$$N=6$$个独立的层堆叠而成的, 除了与Encoder层中的两个完全相同的子层外, 在**两层之间**又加入了一个**multi-head attention**, 这里是对Encoder的输出做attention处理.

与Encoder相同, 每个子层也引入了**residual connection**, 并且**相加之后**使用**layer normalization**得到每个子层最后的输出.

此外, 为了防止序列中元素的位置主导输出结果, 对Decoder层的**multi-head self-attention**层增加了**mask**操作, 并且结合对**output embedding**结果进行**右移一位**的操作, 保证了每个位置$$i$$的输出, 只会依赖于$$i$$位之前(不包括$$i$$位, 因为右移一位和mask).

## Attention

论文中将常用的Attention结构从新的一种角度进行了描述:

**Attention**作为一种函数, 接受的输入为:

- 一个**query**
- 一组**key-value pairs**

即包含三部分, **query**, **keys**和**values**. 三者都是**向量**.

输出就是对组中所有**values**的加权之和, 其中的权值是使用**compatibility function**(如**内积**), 对组内的每一个**keys**和**query**计算得到的.

例如, 对于常见的**self-attention**来说, 这里值的就是对于序列中的某一个元素对应的向量, 求得经过self-attention之后对应的向量. query指的是这个元素对应的向量(如NLP任务中句子序列中某一个单词对应的embedding向量), key-value pairs就是这个序列的所有元素, 其中的每个元素对应的key和value是完全相同的向量, 对于要比较的那个元素, 与query也是完全相同的. 然后使用当前向量和所有向量做内积得到权值, 最后的数据就是这个权值和对应向量的加权和.

论文中使用了两种Attention方法, 分别为**Scaled Dot-Product Attention**和**Multi-Head Attention
Instead**.

![](http://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gWibp1593y9ib5hyUv34YYrkDnOaEYTg1FcozIHx6MFtHTNRHlQLEAYTf9UTudqRvepQTktXq5YkLVXA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

#### Scaled Dot-Product Attention

我们假设`query`和`key`这两个用来比较的向量, 长度都为$$d_k$$; `value`向量的长度为$$d_v$$. 对`query`和所有`keys`进行**点积**得到值, 再对这里得到的每个点积结果**除以**$$\sqrt{d_k}$$, 完成**scale**, 最后应用一个**softmax function**获得每个`value`对应的**权值**, 加权求得最后的输出向量.

这是对于一个`query`的情况. 实际中是直接对一个序列对应的所有`querys`直接进行计算, 将所有`querys`拼接成一个大的$$Q$$矩阵, 对应的`keys`和`values`也拼接成$$K$$和$$V$$矩阵, 则Scaled Dot-Product Attention对应的计算公式为:

$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

**需要注意的点是**: $$d_k$$较大时, 向量之间的点积结果可能就会非常大, 这回造成softmax函数陷入到**梯度很小**的区域. 为了应对这种情况, 适应了缩放因子$$\sqrt{d_k}$$, 将点积结果尽量缩小到梯度敏感的区域内.

#### Multi-Head Attention
之前的方法都是对$$d_{model}$$维度的`querys`, `keys`和`values`直接使用一个Attention函数, 得到结果, 在Multi-Head Attention方法中, 我们如下操作:

- 对`querys`, `keys`和`values`都分别进行$$h$$次的**线性映射**(类似于SVM中的线性核), 得到$$h$$组维度为分别为$$d_k$$, $$d_k$$, $$d_v$$的三种向量.

  需要注意的是, 这$$h$$次映射都是不同的映射, 每次线性映射使用的参数是不相同的, 而且这个映射是**可学习**的, 相当于得到了$$h$$个不同空间(虽然这些空间的维数是相等的)中的**表征**.

- 然后并行的对这$$h$$组维度为分别为$$d_k$$, $$d_k$$, $$d_v$$的`querys`, `keys`和`values`向量执行Attention函数, 每组都产生一个$$d_v$$维的输出结果.
- 最后将这$$h$$个维度为$$d_k$$向量**拼接起来**.
- 通过线性转换还原成$$d_{model}$$维度的向量.

公式表示为:

$$\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1, \cdots, \text{head}_h)W^O$$

其中:

$$\text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$$

$$W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$$, $$W_i^K \in \mathbb{R}^{d_{model} \times d_k}$$, $$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$$, 以及$$W_i^O \in \mathbb{R}^{hd_v \times d_{model}}$$都是**可学习的**线性映射参数. 在论文中超参数的选择为$$h=8$$, 又由于$$d_{model}=512$$, 因此$$d_k=d_v=d_{model}/h=64$$.

因为中间计算的降维, 总体计算的消耗与直接使用Attention函数的消耗相近.

---

#### Transformer模型中Attention使用的特殊点

对于Multi-Head Attention, 在Transformer模型中有三个不同点:

- 在**encoder-decoder attention**层中, 即Encoder和Decoder两者之间的Attention中(对应于Decoder结构中的中间子层部分), `queries`来自于Decoder结构中上一个子层的输出. 这保证了对于**Decoder中的每一个位置,  都能捕获input sequence各个位置的信息**.

- Encoder中对应的是**self-attention**, 对应一个位置上的`query`, `key`, `value`是完全相同的一个向量. 每个位置的输出结果, 都会参考输入的所有位置.
- 相似的, Decoder中第一个子层也是**self-attention**. 因此对于某个位置的元素, 会获取序列中所有序列的信息. 但为了**防止leftward information flow**(左侧信息泄露), 即防止出现**自回归属性**, 我们对这种**Scaled Dot-Product Attention**通过**mask**进行了限制, 屏蔽从第一个元素到当前元素(包含), 然后再进行Attention操作.

## Position-wise Feed-Forward Networks
Encoder和Decoder都含有一个**fully connected feed-forward network**, 特殊的是, 这个网络**分别对每个位置的attention层的输出向量**单独地进行作用. 整个过程包含了**两次线性变换**以及中间夹杂的一次**ReLU激活**:

$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

对于不同位置的线性变换是**完全一样的**, 即使用相同的参数.

这一层的输入输出都是$$d_{model}=512$$, 中间隐层的维度为$$d_{ff}=2048$$.

## Embeddings and Softmax
使用已经训练好的embeddings将input token和output token转换成$$d_{model}$$维度的向量.

在最后Decoder的输出时, 将Decoder的输出通过一层**线性变换层**和一个**softmax层**, 转换成预测下一个token的概率向量. 这两个层中的参数也是提前训练好的.

在模型中, 两个embedding layers以及最后的softmax之前的线性变换, 这**三者**共享使用相同的矩阵权值.

对于embedding层, 里面的权值需要乘以$$\sqrt{d_{model}}$$之后再使用.

## Positional Encoding
因为模型完全没有使用**循环**(RNN)和**卷积**(CNN), 而又想使用序列中的**顺序信息**, 就必须加入一些关于`token`的**相对位置**和**绝对位置**的信息. 因此我们加入**Positional Encoding**, 作为Encoder和Decoder的输入. 需要注意的是Positional Encoding产生的向量的维度为$$d_{model}$$, 与原本的embedding向量维度相同, 从而两者可以被**相加**使用.

对位置进行embedding的方法很多, 有训练方法和指定方法, 本文中, 采用**频率不同的$$\sin$$和$$\cos$$函数:

$$PE_{pos,2i} = \sin(pos/10000^{2i/d_model})$$

$$PE_{pos,2i+i} = \cos(pos/10000^{2i/d_model})$$

其中$$pos$$代表位置, $$i$$代表第$$i$$维. 每个维度对应于不同频率不同的正弦函数. 使用这种方法, 我们认为能够反应**相对位置**中包含的信息, 这是因为: 对于一个固定的偏移量$$k$$, $$PE_{pos+k}$$能表示成$$PE_{pos}$$的**线性函数**.

## Why Self-Attention
之所以使用**Self-Attention**而没有使用循环或卷积的结构, 主要出于以下三点的考虑:

- 每层的计算复杂度
- 计算可以并行的程度
- 对于序列问题, **长序列**是一个难点. Self-Attention方法对于长短序列都有较好的表现. 这是由于我们认为在模型中, **前向**和**后向**传播的**路径越短**, 就更容易学习到其中的关系. 对于循环和卷积, 距离当前较远的位置, 在传播过程中都要经过较长的距离. 但对于Self-Attention结构, 无论两个元素在序列中的相对距离如何, 传播的距离总是相等的.

## 参考资料

- [Attention Is All You Need](http://arxiv.org/abs/1706.03762)
- [Kyubyong/**transformer**](https://github.com/Kyubyong)

