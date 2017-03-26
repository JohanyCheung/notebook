## 简介

**Stanford NLP**是包含了*53种语言*预训练模型的自然语言处理工具包. 英文自然是支持的, 但目前仅支持繁体中文, 不支持简体中文, 所以应用还是有限的.

基于`PyTorch`, 支持多种语言的完整文本分析管道, 包含:
- 分词`tokenize`
- 词性标注`pos`
- 词形归并`lemma`
- 依存关系解析`depparse`

代码地址: [Github](https://github.com/stanfordnlp/stanfordnlp)

使用说明: [StanfordNLP](https://stanfordnlp.github.io/stanfordnlp/)

## 使用方法

### 安装与模型下载

包直接使用`pip`进行安装:

```sh
pip install stanfordnlp
```

在使用之前, 需要下载对应语言已经训练好的模型. 可以在程序中通过`download`函数指定目录进行下载.

```python
import stanfordnlp
stanfordnlp.download("en", "path/to/put_in/model")
```

如果下载不动可以直接从说明文档提供的地址进行下载:
- [Installation & Model Download](https://stanfordnlp.github.io/stanfordnlp/installation_download.html)

(虽然可能还是下载很慢, 但用IDM至少可以断点续传:d)

### 加载模型

创建一个`Pipeline`管道对象, 加载所有模型, 使用的方法为:

```python
stanfordnlp.Pipeline(processors=DEFAULT_PROCESSORS_LIST, lang='en', models_dir=DEFAULT_MODEL_DIR, treebank=None,
                 use_gpu=True, **kwargs)
```

关键的参数有:
- **processors**: 这里指定整个管道包含哪些过程, 根据实际中的需求指定. 默认值为`"tokenize,mwt,pos,lemma,depparse"`, 是一个字符串, 过程之间用逗号隔开. 上面5个过程即这个包提供的所有过程, 分别代表:
  - **tokenize**: 分词
  - **mwt**: 词合并, 意思是多个单词合在一起, 作为一个整体表意.
  - **pos**: 词性标注
  - **lemma**: 词元表示, 即词形归并. 将*时态*, *单复数*等变形还原回词元的形式. 并不会直接代替, 而是通过结果中属性的方式进行保存.
  - **depparse**: 依存关系解析
- **lang**: 语言. 指定处理的语言, 注意要提前下载好对应语言的模型.
- **models_dir**: 模型所处的目录位置.
- **use_gpu**: 是否使用GPU.

创建一个`pipeline`之后, 就可以使用它的`__call__`方法得到结果了.

```python
pipeline = stanfordnlp.Pipeline(models_dir="path/to/put_in/model", lang="en", use_gpu=True)
doc = pipeline("Barack Obama was born in Hawaii.  He was elected president in 2008.")
```

### 结果提取

上面得到的`doc`就是包含所有结果的一个对象. `StanfordNLP`的结果是分为若干的级别的, 每个级别都有自己独特的属性, 以及连接子级结果的属性. 从高到低有:

- **Document**: 调用`pipeline`的`__call__`方法得到的结果就是`Document`. 我们提供的字符串被认为是一篇文章, 因此最高级别的结果就是文章所对应的级别.

    ```python
    doc = pipeline("Barack Obama was born in Hawaii.  He was elected president in 2008.")
    doc
    ```

    ```python
    <stanfordnlp.pipeline.doc.Document at 0x26647dfe518>
    ```

- **Sentence**: 每篇文章是有若干个句子组成的. 调用`Document`的`sentences`属性即可得到所有句子对象组成的列表.

    ```python
    doc.sentences
    ```

    ```python
    [<stanfordnlp.pipeline.doc.Sentence at 0x2678ec509e8>,
     <stanfordnlp.pipeline.doc.Sentence at 0x2678ec50a58>]
    ```

    拿出其中一个句子. 对于句子对象, 有以下的属性:

    - **words**: 获取这个句子中所有单词对象的结果:
        
        ```python
        doc.sentences[0].words
        ```

        ```python
        [<Word index=1;text=Barack;lemma=Barack;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=4;dependency_relation=nsubj:pass>,
         <Word index=2;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=1;dependency_relation=flat>,
          <Word index=3;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;governor=4;dependency_relation=aux:pass>,
         <Word index=4;text=born;lemma=bear;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;governor=0;dependency_relation=root>,
         <Word index=5;text=in;lemma=in;upos=ADP;xpos=IN;feats=_;governor=6;dependency_relation=case>,
         <Word index=6;text=Hawaii;lemma=Hawaii;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=4;dependency_relation=obl>,
         <Word index=7;text=.;lemma=.;upos=PUNCT;xpos=.;feats=_;governor=4;dependency_relation=punct>]
        ```

        可以看到, 是`Word`对象的列表, 按顺序给出了每个单词的索引, 词元, 词性等等信息. 囊括的信息范围是在初始化`Pipeline`对象时指定的.
    - **tokens**: 获取句子中的`Token`对象. 在`StanfordNLP`中, `token`和`word`是不同的概念, 虽然在形式上, 以及对应关系上看来往往是完全相同的, 但`Token`对象和`Word`对象有着不同的属性和方法, 而且如果pipeline中选择了`mwt`, 会对单词合并成词组, 就可能造成两个列表中元素数量的不同.

        ```python
        doc.sentences[0].tokens
        ```

        ```python
        [<Token index=1;words=[<Word index=1;text=Barack;lemma=Barack;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=4;dependency_relation=nsubj:pass>]>,
         <Token index=2;words=[<Word index=2;text=Obama;lemma=Obama;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=1;dependency_relation=flat>]>,
         <Token index=3;words=[<Word index=3;text=was;lemma=be;upos=AUX;xpos=VBD;feats=Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin;governor=4;dependency_relation=aux:pass>]>,
         <Token index=4;words=[<Word index=4;text=born;lemma=bear;upos=VERB;xpos=VBN;feats=Tense=Past|VerbForm=Part|Voice=Pass;governor=0;dependency_relation=root>]>,
         <Token index=5;words=[<Word index=5;text=in;lemma=in;upos=ADP;xpos=IN;feats=_;governor=6;dependency_relation=case>]>,
         <Token index=6;words=[<Word index=6;text=Hawaii;lemma=Hawaii;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=4;dependency_relation=obl>]>,
         <Token index=7;words=[<Word index=7;text=.;lemma=.;upos=PUNCT;xpos=.;feats=_;governor=4;dependency_relation=punct>]>]
        ```

        可以看到`Token`作为`Word`的上一级, 是可能包含多个单词的, 在`Token.words`属性中体现.
    - **dependencies**: 句子中每个单词的依赖关系, 也是以列表的形式体现的.
- **Token**: 句子的下一级就是token了, 这是分词的结果, 一般来说一个token对应一个word, 但由于`mwt`合并的关系, 也可能包含多个word. 使用`Sentence.tokens`属性获取句子的token列表.

    对于某个token, 有以下的属性和方法可以使用.

    - **words**: 这个token包含的word列表, 通常是只有一个元素的列表.

        ```python
        doc.sentences[0].tokens[0].words
        ```

        ```python
        [<Word index=1;text=Barack;lemma=Barack;upos=PROPN;xpos=NNP;feats=Number=Sing;governor=4;dependency_relation=nsubj:pass>]
        ```

        可以看到是元素是`Word`对象.
    
    - **index**: token对应的索引

        ```python
        doc.sentences[0].tokens[0].index
        ```

        ```python
        '1'
        ```

        注意返回的是一个字符串格式的数字.
    
    - **text**: token对应的字符串

        ```python
        doc.sentences[0].tokens[0].text
        ```

        ```python
        'Barack'
        ```

- **Word**: 最底级的对象, 代表一个单词.
  - **pos**, **xpos**: 两个属性是等价的. **treebank-specific part-of-speech**
  - **upos**: **universal part-of-speech**. 与上面的属性都是表示单词的词性, 属于两套标准体系.
  - **dependency_relation**: 依赖关系
  - **governor**: 依赖于第几个word, index是从1开始的
  - **lemma**: 词元
  - **parent_token**: 单词所属的token, 返回的是一个Token对象
  - **text**: word对应的字符串

    ```python
    doc.sentences[0].words[0].pos
    doc.sentences[0].words[0].xpos
    doc.sentences[0].words[0].upos
    doc.sentences[0].words[0].dependency_relation
    doc.sentences[0].words[0].governor
    doc.sentences[0].words[0].index
    doc.sentences[0].words[0].lemma
    doc.sentences[0].words[0].text
    ```

    ```python
    'NNP'
    'NNP'
    'PROPN'
    'nsubj:pass'
    4
    '1'
    'Barack'
    'Barack'
    ```
