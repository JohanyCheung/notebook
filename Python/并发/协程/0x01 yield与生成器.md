## yield与协程

Python中, 可以通过使用yield定义的生成器, 实现协程.

## 生成器

#### 协程生成器的基本行为

首先定义一个生成器:

```python
def simple_coroutine():
    print('-> coroutine started')
    x = yield
    print('-> coroutine received:', x)
```

形式如同定义函数, 如果定义体中含有yield关键字, 则就是一个生成器的定义体. 创建一个生成器对象:

```python
my_coro = simple_coroutine()
my_coro
```

输出为:

```
<generator object simple_coroutine at 0x00000267ED4245C8>
```

**协程有四种状态**:

- GEN_CREATED: 等待开始执行
- GEN_RUNNING: 解释器正在执行
- GEN_SUSPENDED: 在yield表达式处暂停
- GEN_CLOSED: 执行结束

可以使用inspect包中的`inspect.getgeneratorstate`函数查看某个协程的状态:

```python
import inspect

inspect.getgeneratorstate(my_coro)
```

输出为:

```
GEN_CREATED
```

一个刚创建的生成器, 其状态为**GEN_CREATED**, 此时的协程还没有被激活, 需要经过**预激**(prime)操作, **让协程向前执行到第一个yield表达式**, 准备好作为活跃的协程使用. 预激的方法有两种:

- 调用`next(my_coro)`方法
- 使用协程的send方法, 并传递一个None, `my_coro.send(None)`

```python
my_coro.send(None)
```

输出为:

```
-> coroutine started
```

可以看到, 此时的生成器执行到yield的表达式处, 并将控制权交还给了**调用方**.

查看此时协程的状态:

```python
inspect.getgeneratorstate(my_coro)
```

输出为:

```
GEN_SUSPENDED
```

即状态变成了GEN_SUSPENDED, 代表在yield表达式处暂停.

继续调用:

```python
my_coro.send(42)
```

输出为:

```python
-> coroutine received: 42
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-12-7c96f97a77cb> in <module>()
----> 1 my_coro.send(42)

StopIteration: 
```

可以看出:

- send方法输入的参数, 会被赋给yield表达式所在行的左侧的变量, 此处将`42`赋给了参数`x`
- 生成器执行完之后, 抛出`StopIteration`错误. 可以由调用方捕捉这个错误, 进行相应逻辑的处理

查看此时协程的状态:

```python
inspect.getgeneratorstate(my_coro)
```

输出为:

```
GEN_CLOSED
```

生成器, 也即协程已经正常结束了.

---

如果协程没有预激, 就直接使用`send`方法传递参数, 会报如下的错位:

```python
my_coro1 = simple_coroutine()
print(inspect.getgeneratorstate(my_coro))
my_coro1.send(42)
```

输出为:

```python
GEN_CLOSED
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-14-c1f2b4df7ce8> in <module>()
      1 my_coro1 = simple_coroutine()
      2 print(inspect.getgeneratorstate(my_coro))
----> 3 my_coro1.send(42)

TypeError: can't send non-None value to a just-started generator
```

#### 协程向调用方返回结果

生成器在执行到yield表达式暂停时, 会将yield表达式执行后的结果, 返回给调用方. 看下面的例子:

```python
def simple_coro2(a):
    print('-> coroutine started: a =', a)
    b = yield a  # 返回a的值
    print('-> Received: b =', b)
    c = yield a + b  # 返回a+b的值
    print('-> Received: c =', c)

my_coro2 = simple_coro2(14)  # 创建生成器
print(inspect.getgeneratorstate(my_coro2))
next(my_coro2)  # 激活生成器
```

输出为:

```
GEN_CREATED
-> coroutine started: a = 14
14
```

可以看到激活操作, 执行了第一个yield表达式, 将生成器中`a`的值`14`返回给了调用方, 并等待`send`操作传入值. 继续调用`send`方法, 会将send中指定的参数传递给`b`, 向下执行, 将yield表达式`a+b`的值返回:

```python
my_coro2.send(28)
```

输出为:

```
-> Received: b = 28
42
```

结果如同预期, 将`a+b=42`的结果返回. 继续调用, 协程执行完毕.

```python
my_coro2.send(99)
```

输出为

```python
-> Received: c = 99
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-18-977093b924ab> in <module>()
----> 1 my_coro2.send(99)

StopIteration: 
```

整个过程分三步:

- 调用next(my_coro2), 打印`a`的值, 然后执行yield a, 产出数字14, 返回
- 调用my_coro2.send(28), 把28赋值给b, 打印`b`的值, 然后执行 yield a + b, 产生数字42, 返回
- 调用my_coro2.send(99), 把99赋值给c, 打印`c`的值, 产生数字99, 返回

#### 使用装饰器预激协程

新创建的生成器对象, 每次都要手动进行预激, 比较麻烦, 可以用**装饰器**进行预激操作:

```python
from functools import wraps

def coroutinue(func):
    '''
    装饰器： 向前执行到第一个`yield`表达式，预激`func`
    :param func: func name
    :return: primer
    '''
    @wraps(func)
    def primer(*args, **kwargs):
        # 把装饰器生成器函数替换成这里的primer函数；调用primer函数时，返回预激后的生成器。
        gen = func(*args, **kwargs)
        # 调用被被装饰函数，获取生成器对象
        next(gen)  # 预激生成器
        return gen  # 返回生成器
    return primer
```

定义生成器:

```python
@coroutinue
def simple_coro(a):
    a = yield
```

直接创建的生成器对象就已经是激活的状态:

```python
coro = simple_coro(12)
inspect.getgeneratorstate(coro)
```

输出为:

```
GEN_SUSPENDED
```

#### 终止协程和异常处理

**协程向调用方**

协程运行过程中的异常, 会向上冒泡, 传递给`next`函数或者`send`方法的调用方, 调用方如果没有对其进行处理, 就会导致错误终止.

```python
@coroutinue
def averager():
    # 使用协程求平均值
    total = 0.0
    count = 0
    average = None
    while True:
        term = yield average
        total += term
        count += 1
        average = total/count
```

```python
coro_avg = averager()
print(coro_avg.send(40))
print(coro_avg.send(50))
print(coro_avg.send('123')) # 由于发送的不是数字，导致内部有异常抛出。
```

结果为:

```python
40.0
45.0
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-23-0aa2c396e8a8> in <module>()
      2 print(coro_avg.send(40))
      3 print(coro_avg.send(50))
----> 4 print(coro_avg.send('123')) # 由于发送的不是数字，导致内部有异常抛出。

<ipython-input-22-f96454dd016a> in averager()
      7     while True:
      8         term = yield average
----> 9         total += term
     10         count += 1
     11         average = total/count

TypeError: unsupported operand type(s) for +=: 'float' and 'str'
```

**调用方向协程**

上面的例子, 换个角度, 可以看做是**传入特别的参数**, 手动使协程退出. 而生成器中有两个特别的方法可以实现这种事情.

- `throw`方法

  `generator.throw(exc_type[, exc_value[, traceback]])`, 这个方法**使生成器在暂停的yield表达式处抛出指定的异常**, 如果生成器处理了抛出的异常, 代码会继续向下进行, 直到遇到下一个yield表达式或整个代码运行完毕. 当然, 如同`send`方法一样, 下一个yield产出的值会成为调用throw方法得到的返回值. 如果没有处理, 则向上冒泡, 直接抛出

- `close`方法

  `generator.close()`, 生成器在暂停的yield表达式处抛出**GeneratorExit**异常, 如果生成器没有处理这个异常或者抛出了StopIteration异常, 调用方不会报错, 如果收到GeneratorExit异常, 生成器一定不能产出值, 否则解释器会抛出RuntimeError异常.

```python
class DemoException(Exception):
    pass
```

```python
@coroutinue
def exc_handling():
    print('-> coroutine started')
    while True:
        try:
            x = yield
        except DemoException:
            print('*** DemoException handled. Conginuing...')
        else:
            # 如果没有异常显示接收到的值
            print('--> coroutine received: {!r}'.format(x))
    raise RuntimeError('This line should never run.')  # 这一行永远不会执行 
```

```python
exc_coro = exc_handling()

exc_coro.send(11)
exc_coro.send(12)
exc_coro.send(13)
exc_coro.close()
```

输出为:

```
-> coroutine started
--> coroutine received: 11
--> coroutine received: 12
--> coroutine received: 13
```

查看当前状态:

```python
inspect.getgeneratorstate(exc_coro)
```

结果为:

```
GEN_CLOSED
```

---

如果传入DemoException, 因为做了异常处理, 协程不会中止:

```python
exc_coro = exc_handling()

exc_coro.send(11)
exc_coro.send(12)
exc_coro.send(13)
exc_coro.throw(DemoException) # 协程不会中止，但是如果传入的是未处理的异常，协程会终止
```

结果为:

```
-> coroutine started
--> coroutine received: 11
--> coroutine received: 12
--> coroutine received: 13
*** DemoException handled. Conginuing...
```

查看当前状态:

```python
inspect.getgeneratorstate(exc_coro)
```

结果为:

```
GEN_SUSPENDED
```

#### 协程return

由于协程执行完毕后向外报`StopIteration`错误, 对于有`return`值的协程生成器, 如何获取它return的值, 是需要考察的问题.

```python
from collections import namedtuple

Result = namedtuple('Result', 'count average')

def averager():
    total = 0.0
    count = 0
    average = None
    while True:
        term = yield
        if term is None:
            break  # 为了返回值，协程必须正常终止；这里是退出条件
        total += term
        count += 1
        average = total/count    # 返回一个namedtuple，包含count和average两个字段。在python3.3前，如果生成器返回值，会报错
    return Result(count, average)
```

```python
coro_avg = averager()
next(coro_avg)
coro_avg.send(20) # 并没有返回值
coro_avg.send(30)
coro_avg.send(40)
# 发送None终止循环，导致协程结束。生成器对象会抛出StopIteration异常。异常对象的value属性保存着返回值。
coro_avg.send(None)
```

结果为:

```python
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-32-e7656e6f4153> in <module>()
      5 coro_avg.send(40)
      6 # 发送None终止循环，导致协程结束。生成器对象会抛出StopIteration异常。异常对象的value属性保存着返回值。
----> 7 coro_avg.send(None)

StopIteration: Result(count=3, average=30.0)
```

可以看到奇特的一点, return表达式的值, 会作为**StopIteration**异常的一个属性值被带出, 这样做是为了保留生成器对象耗尽时抛出StopIteration异常的行为. 如果要获取这个值, 就需要如下的操作:

```python
coro_avg = averager()
next(coro_avg)
coro_avg.send(20) # 并没有返回值
coro_avg.send(30)
coro_avg.send(40)
try:
    coro_avg.send(None)
except StopIteration as exc:
    result = exc.value
result
```

就能得到结果:

```python
Result(count=3, average=30.0)
```

## 参考资料

[python协程1：协程 10分钟入门](https://gusibi.github.io/post/python-coroutine-1-yield/)

[python协程2：yield from 分析](https://mp.weixin.qq.com/s/AXaD7vhMYBJdSgxx7Gr3_Q)

