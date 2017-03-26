借助[BERT](https://arxiv.org/pdf/1810.04805.pdf)论文, 梳理下自然语言处理当前常见的任务.

## NLP任务

根据判断主题的级别, 将所有的NLP任务分为两种类型:

- **token-level task**: token级别的任务. 如**完形填空**(Cloze), 预测句子中某个位置的单词; 或者**实体识别**; 或是**词性标注**; **SQuAD**等.
- **sequence-level task**: 序列级别的任务, 也可以理解为句子级别的任务. 如**情感分类**等各种句子分类问题; 推断两个句子的是否是同义等.

## token-level task

#### Cloze task

即`BERT`模型预训练的两个任务之一, 等价于**完形填空任务**, 即给出句子中其他的上下午`token`, 推测出当前位置应当是什么`token`.

解决这个问题就可以直接参考`BERT`在预训练时使用到的模型: **masked language model**. 即在与训练时, 将句子中的部分`token`用`[masked]`这个特殊的`token`进行替换, 就是将部分单词遮掩住, 然后目标就是预测`[masked]`对应位置的单词.

这种训练的好处是不需要人工标注的数据. 只需要通过合适的方法, 对现有语料中的句子进行随机的遮掩即可得到可以用来训练的语料. 训练好的模型, 就可以直接使用了.

#### SQuAD(Standford Question Answering Dataset) task

这是一个**生成式**的任务. 样本为语句对. 给出一个问题, 和一段来自于*Wikipedia*的文本, 其中这段文本之中, 包含这个问题的答案, 返回一短语句作为答案.

因为给出答案, 这是一个生成式的问题, 这个问题的特殊性在于最终的答案包含在语句对的文本内容之中, 是有范围的, 而且是连续分布在内容之中的.

因此, 我们找出答案在文本语句的开始和结尾处, 就能找到最后的答案. 通过对文本语句序列中每个token对应的所有**hidden vector**做**softmax**判断是开始的概率和是结束的概率, 最大化这个概率就能进行训练, 并得到输出的结果.

#### Named Entity Recognition

本质是对句子中的每个token打标签, 判断每个token的类别.

常用的数据集有:

- **NER**(Named Entity Recognition) **dataset**: 对应于`Person`,  `Organization`, `Location`, `Miscellaneous`, or `Other (non-named entity)`.

## sequence-level task

#### NLI(Natural Language Inference) task

**自然语言推断任务**, 即给出**一对**(a pair of)句子, 判断两个句子是*entailment*(相近), *contradiction*(矛盾)还是*neutral*(中立)的. 由于也是分类问题, 也被称为**sentence pair classification tasks**.

在智能问答, 智能客服, 多轮对话中有应用.

常用的数据集有:

- **MNLI**(Multi-Genre Natural Language Inference): 是[**GLUE Datasets**](https://gluebenchmark.com/leaderboard)(General Language Understanding Evaluation)中的一个数据集. 是一个大规模的来源众多的数据集, 目的就是推断两个句子是意思相近, 矛盾, 还是无关的.
- **WNLI**(Winograd NLI)

#### Sentence Pair Classification tasks

两个句子相关性的分类问题, `NLI task`是其中的特殊情况. 经典的此类问题和对应的数据集有:

- **QQP**(Quora Question Pairs): 这是一个**二分类**数据集. 目的是判断两个来自于`Quora`的问题句子在语义上是否是等价的.
- **QNLI**(Question Natural Language Inference): 也是一个**二分类**问题, 两个句子是一个`(question, answer)`对. 正样本为`answer`是对应`question`的答案, 负样本则相反.
- **STS-B**(Semantic Textual Similarity Benchmark): 这是一个类似**回归**的问题. 给出一对句子, 使用`1~5`的评分评价两者在语义上的相似程度.
- **MRPC**(Microsoft Research Paraphrase Corpus): 句子对来源于对同一条新闻的评论. 判断这一对句子在语义上是否相同.
- **RTE**(Recognizing Textual Entailment): 是一个**二分类**问题, 类似于**MNLI**, 但是数据量少很多.

#### Single Sentence Classification tasks

- **SST-2**(Stanford Sentiment Treebank): 单句的**二分类**问题, 句子的来源于人们对一部电影的评价, 判断这个句子的情感.
- **CoLA**(Corpus of Linguistic Acceptability): 单句的**二分类**问题, 判断一个英文句子在语法上是不是可接受的.

#### SWAG(Situations With Adversarial Generations)

给出一个陈述句子和4个备选句子, 判断前者与后者中的哪一个最有**逻辑的连续性**, 相当于**阅读理解**问题.

