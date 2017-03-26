`Python`中所有变量皆为对象. 有对象, 就有对象的类别. 代码中可以使用内置函数`isinstance(obj, type)`判断一个对象`obj`是否是`type`类的对象.

常见的类这里就不列举了, 这里记录一些比较巧妙和少见的类别判定方法, 在实际开发中也能经常被使用到.

### 判断是否是类

```python
isinstance(obj, type)
```

### 判断是否是函数

这里的函数指的是在模块中使用直接定义的函数, 以及**类函数**. 注意, **对象的函数称为方法(method)**, 不在此类中.

```python
isinstance(obj, types.FunctionType)
```

### 判断是否是方法

类对象的函数称为方法(method).

```python
isinstance(obj, types.MethodType)
```

### 判断是否可迭代

```python
isinstance(obj, collections.Iterable)
```

或者使用:

```python
hasattr(obj, "__iter__")
```

### 判断是否是迭代器

```python
isinstance(obj, collections.Iterator)
```

或者使用:

```python
hasattr(obj, "__next__")
```
