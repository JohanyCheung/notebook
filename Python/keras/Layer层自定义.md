keras允许自定义Layer层, 大大方便了一些复杂操作的实现. 也方便了一些novel结构的复用, 提高搭建模型的效率.

## 实现方法

通过继承`keras.engine.Layer`类, 重写其中的部分方法, 实现层的自定义. 主要需要实现的方法及其意义有:

- _ _init_ _(self, **kwargs)

  作为类的初始化方法, 一般将需要传入的自定义参数存为对象的属性. 需要注意的有以下几点:

  - 由于继承Layer类, 所以在处理完自定义的参数之后, 仍可能还有参数需要父类处理, 所以需要调用父类的初始化方法, 将kwargs参数传入:

    ```python
    class DecayingDropout(Layer):
        def __init__(self, initial_keep_rate=1., decay_interval=10000, decay_rate=0.977, noise_shape=None, seed=None, **kwargs):
            super(DecayingDropout, self).__init__(**kwargs)
    ```

  - 对象的`self.supports_masking`方法的作用是本层中是否涉及到使用mask或对mask矩阵进行计算. mask的作用是屏蔽传入Tensor的部分值, 常常在NLP问题中, 对句子padding之后, 不想让填补的0值对应的位置参与运算而使用. 这个参数默认为False, 如果有使用到, 需要将其值为True:

    ```python
    self.supports_masking = True
    ```

- `build(self, input_shape, **kwargs)`

  这里是定义权重的地方, 需要注意的有以下几点:

  - 通过`self.add_weight`方法定义权重, 且需要将权重存为类的属性, 例如:

    ```python
    self.iterations = self.add_weight(name='iterations',
                                      shape=(1,),
                                      dtype=K.floatx(),
                                      initializer='zeros',
                                      trainable=False)
    ```

    其中`self.iterations`需要在初始化时设置为None, 符合类编程的习惯. `self.add_weight`方法有若干参数, 常用的即为上面几个.

  - 由于要求build方法必须设置`self.built = True `, 而这个方法在父类中实现, 因此, 在方法的**最后**需要调用:

    ```python
    super(DecayingDropout, self).build(input_shape)
    ```

- `call(self, inputs, **kwargs)`

  这里是编写层的功能逻辑的地方, 传入的第一个参数即输入张量, 即调用_ _call_ _方法传入的张量. 除此之外, 需要注意的点有:

  - 如果需要在计算的过程中使用mask, 则需要传入mask参数:

    ```python
    def call(self, x, mask=None):
            if mask is not None:
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0,2,1])
                mask = K.cast(mask, K.floatx())
                x = x * mask
                return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
            else:
                return K.mean(x, axis=self.axis)
    ```

  - 如果该层在训练和预测时的行为不一样(如Dropout)函数, 需要传入指定参数`training`, 即使用布尔值指定调用的环境. 例如在`Dropout`层的源码中, call方法是这样实现的:

    ```python
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
    ```

    `K.in_train_phase() `方法就是用来区别在不同环境调用时, 返回的不同值的. 这个函数通过`training`参数区别调用环境, 如果是训练环境, 则返回第一个参数对应的结果, 预测环境则返回第二个参数对应的结果. 可以传入函数, 返回这个函数对应的返回结果.

  - 除了计算之外, 这个函数也是更新层内参数的地方, 即build方法中增加的参数. 通过`self.add_update`方法进行更新, 例如:

    ```python
    def call(sekf, x):
        self.add_update([K.moving_average_update(self.moving_mean, mean,self.momentum),
                         K.moving_average_update(self.moving_variance,variance,self.momentum)],
                        inputs)
    ```

    或者:

    ```python
    def call(self, inputs, training=None):
        self.add_update([K.update_add(self.iterations, [1])], inputs)
    ```

    可以看到, `self.add_update`方法传入一个列表, 包含一些列更新的动作. 这些更新的动作需要借助`K`的一些函数实现, 如`K.moving_average_update`, `K.update_add`等等.

    另外还可以传入`inputs`函数, 作为更新的前提条件.

---

除此之外, 还有一些常常需要重新定义的方法:

- `get_config(self)`:

  返回层的一些参数. 对于自定义的参数, 需要在此指定返回:

  ```python
  def get_config(self):
      config = {'initial_keep_rate':  self.initial_keep_rate,
                'decay_interval':     self.decay_interval,
                'decay_rate':         self.decay_rate,
                'noise_shape':        self.noise_shape,
                'seed':               self.seed}
      base_config = super(DecayingDropout, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))
  ```

- `compute_output_shape(input_shape) `:

  计算输出shape. input_shape是输入数据的shape.

- `compute_mask(self, input, input_mask=None)`:

  计算输出的mask, 其中input_mask为输入的mask. 需要注意的有:

  - 如果input_mask为None, 说明上一层没有mask. 可以在本层创建一个新的mask矩阵.

  - 如果以后的层不需要使用mask, 返回None即可, 之后就不存在mask矩阵了

    ```python
    def compute_mask(self, input, input_mask=None):
            # need not to pass the mask to next layers
            return None
    ```

  - 如果经过本层, mask矩阵没有变化, 不用实现该函数, 只需要在初始化时, 指定`self.supports_masking = True`即可.

## 参考资料

[编写你自己的Keras层](https://keras.io/zh/layers/writing-your-own-keras-layers/)

[Keras编写自定义层--以GroupNormalization为例](https://zhuanlan.zhihu.com/p/36436904)

[Keras自定义实现带masking的meanpooling层](https://blog.csdn.net/songbinxu/article/details/80148856)

[Keras实现支持masking的Flatten层](https://blog.csdn.net/songbinxu/article/details/80254122)
