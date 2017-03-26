### K-cores

算法地址: [An O(m) Algorithm for Cores Decomposition of Networks](https://arxiv.org/abs/cs/0310049). 论文的摘要中提到:

```
The structure of large networks can be revealed by partitioning them to smaller parts, which are easier to handle. One of such decompositions is based on k--cores, proposed in 1983 by Seidman. In the paper an efficient, O(m), m is the number of lines, algorithm for determining the cores decomposition of a given network is presented.
```

因此, 这是一种将**图分解成若干部分的算法**. 模块中相关的方法有:

#### 1. nx.k_core(*G*, *k=None*, *core_number=None*) 

作用: 返回图`G`的`k-core`, `k-core`是包含所有大于或等于度(`degree`)`k`的点组成的最大子图, 这里的度指的是入度+出度.

返回: 子图对应的`Graph`对象.

注意: 

- 传入的图中不能有`self loops `或`parallel edges `.
- 度指的是最后得到的最大子图中所有点的度符合条件, 而不是指原图中结点的度, 否则就只是筛选问题了.

#### 2. nx.k_shell(*G*, *k=None*, *core_number=None*) 

与`k_core`相似, 但是`k-shell`是所有度恰好为`k`的点组成的子图, 而且是从`k-core`子图中得到的. 函数的参数, 返回形式与注意点都与`k_core`相似.

#### 3. core_number(*G*)

作用: 返回每个点的`core number`. 每个点可能都对应若干个不同`k`大小的`k-core`, 某个点的`core number`指的就是这个点最大的`k`值.

返回: 返回值是字典类型, `key`是`node`, `value`是对应的`k`值.

**关于`k-core`算法的使用**可以参考这篇文章: [NetworkX学习笔记 - 基本功能使用](https://blog.csdn.net/SunCherryDream/article/details/53234254?locationNum=2&fps=1)



