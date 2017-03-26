## BiLSTM-CRF模型

### 论文参考

- [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)

### 原理

#### Look-up层

以句子为单位, 将一个含有$$n$$个字的句子记作:

$$x=(x_1,\cdots,x_n)$$

$$x_i$$代表这这个字对应在字典中的**ID**. 使用训练好的embedding矩阵将句子中的每个字映射为embedding向量, 假设向量是$$d$$维的, $$\boldsymbol x_{i}\in\mathbb R^{d}$$. 这就是整个网络的第一层, **Look-up层**. 可以在输入到下一层之前加入`dropout`层加强鲁棒性.

#### 双向LSTM层

模型的第二层就是**双向LSTM层**, 用来**自动提取句子特征**. 将一个句子的各个字的**char embedding**序列$$(\boldsymbol x_{1},\boldsymbol x_{2},...,\boldsymbol x_{n})$$作为双向LSTM各个时间步的输入, 再将正向LSTM输出的隐状态序列$$(\overset{\longrightarrow}{\boldsymbol h_1},\overset{\longrightarrow}{\boldsymbol h_2},...,\overset{\longrightarrow}{\boldsymbol h_n})$$与反向LSTM的隐状态序列$$(\overset{\longleftarrow}{\boldsymbol h_1},\overset{\longleftarrow}{\boldsymbol h_2},...,\overset{\longleftarrow}{\boldsymbol h_n})$$在各个位置进行拼接$$\boldsymbol h_{t}=[\overset{\longrightarrow}{\boldsymbol h_t};\overset{\longleftarrow}{\boldsymbol h_t}]\in\mathbb R^{m}$$, 得到完整的隐状态序列:

$$({\boldsymbol h_1},{\boldsymbol h_2},...,{\boldsymbol h_n})\in\mathbb R^{n\times m}$$

传入到`dropout`层后, 接入一个**线性层**, 将隐状态向量从$$m$$维映射到$$k$$维, $$k$$是**标注集的标签数**. 这个线性层, 对每个时间片**单独作用**(Time Distribute), 并且所有时间片使用的参数是一样的, 经过这一步后得到:

$$P=({\boldsymbol p_1},{\boldsymbol p_2},...,{\boldsymbol p_n})\in\mathbb R^{n\times k}$$

可以把$$\boldsymbol p_i\in\mathbb R^{k}$$中的每一维$$p_{ij}$$都视为字$$x_{i}$$对第$$j$$个**标签**的得分(非标准概率).

这个线性层类似于**softmax**层, 但只是算出了每类的数值, 并没有做**softmax**函数作用, 也没有判定分类. 而是输入到下一层中继续使用.

#### CRF层

CRF层进行句子级的序列标注.

CRF层的参数是一个$$(k+2)\times (k+2)$$的矩阵, 记为$$A$$, $$A_{ij}$$表示的是第$$i$$个标签到第$$j$$个标签的转移得分, 进而在为一个位置进行标注的时候可以利用此前已经标注过的标签. 之所以要加2是因为要为句子首部添加一个起始状态以及为句子尾部添加一个终止状态.

标签序列$$y=(y_1,y_2,...,y_n)$$, 模型对于句子$$x$$的标签$$y$$的总分数等于各个位置的打分之和:

$$score(x,y)=\sum\limits_{i=1}^{n}P_{i,y_{i}}+\sum\limits_{i=1}^{n+1}A_{y_{i-1},y_{i}}$$

每个位置的打分由两部分得到一部分是由LSTM输出的$$\boldsymbol p_i$$决定, 另一部分则由CRF的转移矩阵$$A$$决定. 进而可以利用Softmax得到归一化后的概率:

$$P(y|x)=\frac{\exp(score(x,y))}{\sum\limits_{y'}\exp(score(x,y'))}$$

模型通过对数似然函数最大化求解得到. 对一个训练样本$$(x,y_{x})$$, 对数似然为:

$$\log P(y_{x}|x)=score(x,y_{x})-\log(\sum_{y'}\exp(score(x,y')))$$

### 预测

模型在预测过程解码时使用动态规划的**Viterbi算法*来求解最优路径.

$$y^{*}=\arg\max_{y'}score(x,y')$$

### 整体结构

整个模型的结构如下图所示:

![](pics/1008922-20170726163008156-1916269133.png)

## 参考资料

论文:

- [Bidirectional LSTM-CRF Models for Sequence Tagging](http://arxiv.org/abs/1508.01991)
- [Neural Architectures for Named Entity Recognition](http://arxiv.org/abs/1603.01360)
- [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://arxiv.org/abs/1603.01354)

解析:

- [序列标注：BiLSTM-CRF模型做基于字的中文命名实体识别](http://www.cnblogs.com/Determined22/p/7238342.html)
- [Sequence Tagging with Tensorflow](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)

代码:

- [A simple BiLSTM-CRF model for Chinese Named Entity Recognition](https://github.com/Determined22/zh-NER-TF), 简单的中文NER例子
- [macanv/BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER), 使用**BERT**进行BER任务, 中文
- [FanhuaandLuomu/BiLstm_CNN_CRF_CWS(法律文档合同类案件领域分词)](https://github.com/FanhuaandLuomu/BiLstm_CNN_CRF_CWS)
- [UKPLab/emnlp2017-bilstm-cnn-crf](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf)
