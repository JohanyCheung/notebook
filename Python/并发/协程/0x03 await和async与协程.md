## 协程与生成器

之前是用生成器来实现的协程的功能, 但其本质还是一个生成器. 这样容易与作为只作为生成器使用的情况, 容易出现混淆的情况. 为了避免这种情况, 需要在定义时与生成器加以区别.

有两种方法:

- **基于生成器的协程**: 使用**asyncio.coroutine**或**types.coroutine**装饰器来标注协程
- **原生协程**: 以**async def**定义的协程

**基于生成器的协程定义如下**:

```python
import types
import asyncio

@types.coroutine
def cor2():
    yield
@asyncio.coroutine
def cor3():
    yield

c2 = cor2()
c3 = cor3()
c2, c3
```

输出为:

```python
<generator object cor2 at 0x00000236C91DA728>, <generator object cor3 at 0x00000236C91DA6D0>
```

可以看出仍是生成器对象, 只是这种定义的方法让这种生成器区别于专门的生成器.

**原生协程的定义如下**:

```python
@types.coroutine
def generator_coroutine():
    yield 1
 
async def native_coroutine():
    await generator_coroutine()
```

分别创建下:

```python
c1 = generator_coroutine()
c2 = native_coroutine()
c1, c2
```

结果为:

```python
(<generator object generator_coroutine at 0x00000236C91DA410>,
 <coroutine object native_coroutine at 0x00000236C91DA468>)
```

可以看到两者的区别. 与基于生成器定义的协程在定义上的区别在于:

- 定义时使用**async**声明是原生协程
- 在定义体内不能使用*yield*, 而是使用**await**进行代替

使用方式仍是一样的, 如使用`send`, `throw`, `close`等方法.

另外两者的区别还在于:

- **原生协程**定义里不能用*yield*或*yield from*表达式
- 原生协程没有**_ _ iter _ _**和**_ _ next _ _**方法, 而是使用**_ _ await _ _**方法.

- 基于生成器的协程中不能使用**yield from**原生协程, 原因在于上面一条
- 反过来, 原生协程可以使用**await**基于生成器的协程

## 原生协程的调用

原生协程的调用有两种方法:

- **await**: 使用await表达式来调用协程, 用法和*yield from*类似, 因此只能再**原生协程内部**使用, 且只能接受**awaitable**对象. awaitable对象就是其**_ _ await _ _**对象返回一个迭代器对象.
  - 原生协程和基于生成器的协程都是awaitable对象
- **send**方法: 和基于生成器的协程一样, 可以调用协程对象的send方法进行调用, 用在非协程函数的定义里.

因此整个调用代码的结构类似于:

```python
@types.coroutine
def generator_coroutine():
    yield 1
 
async def native_coroutine():
    await generator_coroutine()
 
def main():
    native_coroutine().send(None)
```

## 原生协程的问题

#### 原生协程不能真正的暂停执行并强制性返回给事件循环

假设事件循环在`main`函数里, 原生协程是`native_coroutine`, 怎么做才能让原生协程暂停, 并返回到`main`函数里?

`native_coroutine`里面不能使用`yield`, 只能使用`await`, 这只会进入到更深的协程里. 因此只能使用`return`或`raise`方法返回到`main`, 但这种方法不是暂停, 而是退出, 因此再也回不到`native_coroutine`协程里了.

因此, 对于异步编程, 最外层的事件循环(`main`)如果需要调用协程`send`方法, 则大部分的异步方法都可以用原生协程来实现, 但**最底层的异步方法则需要用基于生成器的协程**.

## 参考资料

[用 Python 3 的 async / await 做异步编程](https://www.keakon.net/2017/06/28/%E7%94%A8Python3%E7%9A%84async/await%E5%81%9A%E5%BC%82%E6%AD%A5%E7%BC%96%E7%A8%8B)

