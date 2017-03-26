**Series.str**是处理字符串列的方法, 拥有的方法与python中原有的字符串`str`对象的方法非常类似, 只是个别的方法不同, 下面记录一些常用的重要方法.

### Series.str.cat

```python
Series.str.cat(others=None, sep=None, na_rep=None, join=None)
```

作用是将两个Series对应位置进行合并, 要求两者的长度相同.

- others: Series, Index, DataFrame, np.ndarrary or list-like
  - 给出的类列表对象长度应该与调用者的长度相同, 然后将两个列表对应索引位置的字符串合并, 组成新的Series
  - 如果没有给出`others`, 则将调用者的所有字符串合并在一起, 返回一个字符串

    ```python
    >>> s = pd.Series(['a', 'b', np.nan, 'd'])
    >>> s.str.cat(sep=' ')
    'a b d'
    ```
- sep: str, default ‘’
  - 分隔符
- na_rep: str or None, default None
  - 用来代替`missing values`的字符
  - 如果是`None`, 将`missing value`从最终结果中移除
- join: {‘left’, ‘right’, ‘outer’, ‘inner’}, default None
  - 需要调用者与传入的对象都是拥有`index`的`Series`, 然后根据`index`进行`merge`操作

---

例子:

```python
>>> s = pd.Series(['a', 'b', np.nan, 'd'])
>>> s.str.cat(sep=' ')
'a b d'

>>> s.str.cat(sep=' ', na_rep='?')
'a b ? d'
```

```python
>>> s.str.cat(['A', 'B', 'C', 'D'], sep=',')
0    a,A
1    b,B
2    NaN
3    d,D
dtype: object

>>> s.str.cat(['A', 'B', 'C', 'D'], sep=',', na_rep='-')
0    a,A
1    b,B
2    -,C
3    d,D
dtype: object

>>> s.str.cat(['A', 'B', 'C', 'D'], na_rep='-')
0    aA
1    bB
2    -C
3    dD
dtype: object
```

```python
>>> t = pd.Series(['d', 'a', 'e', 'c'], index=[3, 0, 4, 2])
>>> s.str.cat(t, join='left', na_rep='-')
0    aa
1    b-
2    -c
3    dd
dtype: object
>>>
>>> s.str.cat(t, join='outer', na_rep='-')
0    aa
1    b-
2    -c
3    dd
4    -e
dtype: object
>>>
>>> s.str.cat(t, join='inner', na_rep='-')
0    aa
2    -c
3    dd
dtype: object
>>>
>>> s.str.cat(t, join='right', na_rep='-')
3    dd
0    aa
4    -e
2    -c
dtype: object
```

### Series.str.split / Series.str.rsplit

```python
Series.str.split(pat=None, n=-1, expand=False)
Series.str.rsplit(pat=None, n=-1, expand=False)
```

对Series中的每个字符串按指定的分割符进行分割, 注意可以直接扩展成**DataFrame**.

**split**/**rsplit**一个是从左向右, 一个从右向左分割.

- pat: str, optional
  - 分割符, 如果不指定, 则默认为`whitespace`
- n: int, default -1 (all)
  - 限定每个元素划分的次数, 即最多返回`n+1`个元素, 如果指定为`0`, `-1`或者`None`则返回所有结果
  - 满足最大分割次数之后, 剩余的结果将不再分割, 直接作为最后的结果返回
- expand: bool, default False
  - 是否将分割的结果扩展到多个列中, 组成DataFrame
  - **True**: 返回扩展后的**DataFrame**/MultiIndex
  - **False**: 返回的仍是**Series**/Index

---

```python
>>> s = pd.Series(["this is a regular sentence",
"https://docs.python.org/3/tutorial/index.html", np.nan])

>>> s.str.split()
0                   [this, is, a, regular, sentence]
1    [https://docs.python.org/3/tutorial/index.html]
2                                                NaN
dtype: object
```

```python
>>> s.str.split(n=2)
0                     [this, is, a regular sentence]
1    [https://docs.python.org/3/tutorial/index.html]
2                                                NaN
dtype: object

>>> s.str.rsplit(n=2)
0                     [this is a, regular, sentence]
1    [https://docs.python.org/3/tutorial/index.html]
2                                                NaN
dtype: object
```

```python
>>> s.str.split(pat = "/")
0                         [this is a regular sentence]
1    [https:, , docs.python.org, 3, tutorial, index...
2                                                  NaN
dtype: object
```

```python
>>> s.str.split(expand=True)
                                               0     1     2        3
0                                           this    is     a  regular
1  https://docs.python.org/3/tutorial/index.html  None  None     None
2                                            NaN   NaN   NaN      NaN 
             4
0     sentence
1         None
2          NaN

>>> s.str.rsplit("/", n=1, expand=True)
                                    0           1
0          this is a regular sentence        None
1  https://docs.python.org/3/tutorial  index.html
2                                 NaN         NaN
```

### Series.str.join

```python
Series.str.join(sep)
```

与**Series.str.split / Series.str.rsplit**的作用相反, 接受每个元素为一个列表, 将列表与连接符组合在一起形成一个字符串.

- sep: 分隔符

**注意**: 如果列表中任何一个元素为**非字符串**, 最后返回的结果为**NaN**. 详情见下面的例子.

---

```python
>>> s = pd.Series([['lion', 'elephant', 'zebra'],
...                [1.1, 2.2, 3.3],
...                ['cat', np.nan, 'dog'],
...                ['cow', 4.5, 'goat'],
...                ['duck', ['swan', 'fish'], 'guppy']])
>>> s
0        [lion, elephant, zebra]
1                [1.1, 2.2, 3.3]
2                [cat, nan, dog]
3               [cow, 4.5, goat]
4    [duck, [swan, fish], guppy]
dtype: object

>>> s.str.join('-')
0    lion-elephant-zebra
1                    NaN
2                    NaN
3                    NaN
4                    NaN
dtype: object
```

### [Series.str.get_dummies](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.get_dummies.html#pandas.Series.str.get_dummies)

```python
Series.str.get_dummies(sep='|')
```

与上面的`split`方法类似, 但此时的输入的每个元素虽然是有分隔符的字符串, 但所有元素分隔出来的结果集合是有限的, 用来做`One-Hot`.

- sep: string, default “|”
  - 分隔符

---

```python
>>> pd.Series(['a|b', 'a', 'a|c']).str.get_dummies()
   a  b  c
0  1  1  0
1  1  0  0
2  1  0  1

>>> pd.Series(['a|b', np.nan, 'a|c']).str.get_dummies()
   a  b  c
0  1  1  0
1  0  0  0
2  1  0  1
```

### [Series.str.contains](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html#pandas.Series.str.contains)

```python
Series.str.contains(pat, case=True, flags=0, na=nan, regex=True)
```

判断是否包含指定的内容, 传入的可以是一个字符串, 也可以使用**正则表达式**

- pat: str
  - 字符串或正则表达式
- case: bool, default True
  - 是否关心大小写
- regex: bool, default True
  - 是否使用正则表达式
- flags: int, default 0 (no flags)
  - 正则表达式的模式, **re**模块使用, 例如`re.IGNORECASE`
- na: default NaN
  - 填补缺失位置的值

---

```python
>>> ind = pd.Index(['Mouse', 'dog', 'house and parrot', '23.0', np.NaN])
>>> ind.str.contains('23', regex=False)
Index([False, False, False, True, nan], dtype='object')

>>> s1.str.contains('oG', case=True, regex=True)
0    False
1    False
2    False
3    False
4      NaN
dtype: object

>>> s1.str.contains('house|dog', regex=True)
0    False
1     True
2     True
3    False
4      NaN
dtype: object
```

```python
>>> s2 = pd.Series(['40','40.0','41','41.0','35'])
>>> s2.str.contains('.0', regex=True)
0     True
1     True
2    False
3     True
4    False
dtype: bool
```
