## 使用背景

最常见的一种情况, 在`NLP`问题的句子补全方法中, 按照一定的长度, 对句子进行填补和截取操作. 一般使用`keras.preprocessing.sequence`包中的`pad_sequences`方法, 在句子前面或者后面补0. 但是这些零是我们不需要的, 只是为了组成可以计算的结构才填补的. 因此计算过程中, 我们希望用`mask`的思想, 在计算中, 屏蔽这些填补0值得作用. keras中提供了`mask`相关的操作方法.

## 原理

在keras中, `Tensor`在各层之间传递, `Layer`对象接受的上层`Layer`得到的`Tensor`, 输出的经过处理后的`Tensor`.

keras是用一个`mask`矩阵来参与到计算当中, 决定在计算中屏蔽哪些位置的值. 因此`mask`矩阵其中的值就是`True/False`, 其形状一般与对应的`Tensor`相同. 同样与`Tensor`相同的是, `mask`矩阵也会在每层`Layer`被处理, 得到传入到下一层的`mask`情况.

## 使用方法

1. 最直接的, 在`NLP`问题中, 对句子填补之后, 就要输入到`Embedding`层中, 将`token`由`id`转换成对应的`vector`. 我们希望被填补的0值在后续的计算中不产生影响, 就可以在初始化`Embedding`层时指定参数`mask_zero`为`True`, 意思就是屏蔽0值, 即填补的0值.

   在`Embedding`层中的`compute_mask`方法中, 会计算得到`mask`矩阵. 虽然在`Embedding`层中不会使用这个`mask`矩阵, 即0值还是会根据其对应的向量进行查找, 但是这个`mask`矩阵会被传入到下一层中, 如果下一层, 或之后的层会对`mask`进行考虑, 那就会起到对应的作用.

2. 也可以在`keras.layers`包中引用`Masking`类, 使用`mask_value`指定固定的值被屏蔽. 在调用`call`方法时, 就会输出屏蔽后的结果.

   需要注意的是`Masking`这种层的`compute_mask`方法, 源码如下:

   ```python
   def compute_mask(self, inputs, mask=None):
       output_mask = K.any(K.not_equal(inputs, self.mask_value), axis=-1)
       return output_mask
   ```

   可以看到, 这一层输出的`mask`矩阵, 是根据这层的输入得到的, 具体的说是会比输入第一个维度, 这是因为最后一个维度被`K.any(axis=-1)`给去掉了. 在使用时需要注意这种操作的意义以及维度的变化.

## 自定义使用方法

更多的, 我们还是在自定义的层中, 需要支持`mask`操作, 因此需要对应的逻辑.

---

首先, 如果我们希望自定义的这个层支持`mask`操作, 就需要在`__init__`方法中指定:

```python
self.supports_masking = True
```

如果在本层计算中需要使用到`mask`, 则`call`方法需要多传入一个`mask `参数, 即:

```python
def call(self, inputs, mask=None):
    pass
```

然后, 如果还要继续输出mask, 供之后的层使用, 如果不对`mask`矩阵进行变换, 这不用进行任何操作, 否则就需要实现`compute_mask`函数:

```python
def compute_mask(self, inputs, mask=None):
    pass
```

这里的`inputs`就是输入的`Tensor`, 与`call`方法中接收到的一样, `mask`就是上层传入的`mask`矩阵.

如果希望`mask`到此为止, 之后的层不再使用, 则该函数直接返回`None`即可:

```python
def compute_mask(self, inputs, mask=None):
    return None
```

## 参考资料

[Keras自定义实现带masking的meanpooling层](https://blog.csdn.net/songbinxu/article/details/80148856)

[Keras实现支持masking的Flatten层](https://blog.csdn.net/songbinxu/article/details/80254122)

