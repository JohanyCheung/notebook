`networkx`是`python`中处理图数据结构常用的一个模块, 使用简单, 常用的对图计算的需求都已经使用高效的算法实现.

本文主要内容为`networkx`模块的基本使用方法, 包括一些常用的算法需求.

### 无向图

#### 1. 创建无向图

创建一个无点无边的空图:

```python
import networkx as nx
g = nx.Graph()
```

点(nodes/vertices)可以是任何哈希的对象, `int`, `string`, `tuple`等等.

#### 2. 点的添加

图的元素扩张可以通过两种形式: 添加点或添加边.

首先, 可以通过以下的方式向图中添加结点:

```python
g.add_node(1)
```

也可以传入`list`, 添加一系列的点:

```python
g.add_nodes_from([
    "a", "b", "c",
])
```

#### 3. 边的添加

图也可以通过增加边来扩展.

添加一条边:

```python
g.add_edge(1, 2)
```

添加多条边:

```python
g.add_edges_from([(1,2),(1,3)])
```

#### 4. 点和边的删除

同样的, 可以便捷的使用方法, 删除图中已经存在的点和边:

```py
Graph.remove_node()
Graph.remove_nodes_from()
Graph.remove_edge()
Graph.remove_edges_from()
```

#### 5. 查看图的属性

使用一些方法快速查询图的一些属性:

```pyt
g.number_of_nodes() # 图中结点的数量
g.number_of_edges() # 图中边的数量

g.nodes() # 图中所有的结点, 以列表形式返回
g.edges() # 图中所有边

g.neighbors(node) # 传入一个node, 返回与这个node相邻所有结点
```

`networkx`还提供了`iterator`的方法来快速定位点和边, 但需要注意的是**不能修改返回的字典格式的数据**, 因为这是模块组织数据的形式. 

```python
>>> g[1]  # 以字典的形式, 返回结点1相连的结点, 以及他们之间边的的属性
{2: {}}

>>> g[1][2] # 返回结点1和结点2之间的边(要求两点相连), 以字典的形式返回, 字典的内容为边的一些属性
{}

>>> g[1][3]['color']='blue'  # 可以这样来定义点1和点3之间边的color属性
```

#### 6. 设置点, 边, 图的属性

```python
>>> g.graph # 通过graph属性获取图的属性, 属性是通过字典的形式表示的
>>> g.graph['day']='Monday' # 设置图的某个属性
```

```python
>>> g.node # 通过node属性获取所有点, 以字典的形式表示
>>> g.node[1] # 获取结点1, 返回的是结点1的属性, 以字典的形式表示
{'time': '5pm'}

# 设置属性
>>> g.add_node(1, time='5pm') # 可以在创建点的时候设置
>>> node[1]['room'] = 714 # 可以在创建点之后设置

>>> g.nodes(data=True) # 返回所有结点, 将data参数设置为True, 则返回每个点的属性数据
[(1, {'room': 714, 'time': '5pm'}), (3, {'time': '2pm'})]
```

```python
# 可以通过以下多种方式设置边的属性
g.add_edge(1, 2, weight=4.7)
g.add_edges_from([(3,4),(4,5)], color='red')
g.add_edges_from([(1,2,{'color':'blue'}), (2,3,{'weight':8})])
g.edge[1][2]['weight'] = 4
g[1][2]['weight'] = 4.7
```

### 有向图

#### 1. 创建有向图

```python
dg = nx.DiGraph()
```

#### 2. 添加带权重的边

这里的每个元素位置要固定, 最后一个元素表示边的权值

```python
dg.add_weighted_edges_from([(1,2,0.5), (3,1,0.75)])
```

#### 3. 点属性

```python
>>> dg.out_degree(1, weight='weight') # 点1的出度
0.5
>>> dg.in_degree(1, weight="weight") # 点1的入度
0.75
>>> dg.degree(1, weight='weight') # 点1的度, 入度 + 出度
1.25
>>> dg.successors(1) # 点1的下一个结点
[2]
>>> dg.neighbors(1) # 点1的相邻结点
[2]
```

#### 4. 转为无向图

```python
g = nx.Graph(dg)
```

