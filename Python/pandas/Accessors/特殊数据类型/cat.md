## 类型转换

**cat**这个访问器是对pandas中的类别类型的列进行特定的操作. 使用前, 需要将对应的列转换成类别类型.

创建时指定为`category`格式:

```python
In [1]: s = pd.Series(["a", "b", "c", "a"], dtype="category")

In [2]: s
Out[2]: 
0    a
1    b
2    c
3    a
dtype: category
Categories (3, object): [a, b, c]
```

---

对于已经存在的`Series`, 可以使用`Series.astype("category")`的方法将原有类型转换成类别类型.

## 特别地

由于类别类型在底层是以**短整型**的格式来存储的, 因此如果原数据类型是**字符串**表示的类别, 转换后将会**大大地减少内存的占用**, 且**后续的操作也会更高效**, 因此将代表类别的列`Series`转换成类别类型是一种相当好的方法, 注意在数据处理过程中经常使用.

## 操作

关于**cat**的操作详见[cat](http://pandas.pydata.org/pandas-docs/stable/reference/series.html#api-series-cat). 介绍一些重要的方法.

#### CateSeries.cat.codes

在转换成类别类型之后, 每个类别在底层都有一个唯一的数值与其相对应(数值范围在`[0, #categories)`之间. 使用这个方法就能得到类别的数值表示方法.
