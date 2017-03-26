## yield from

**yield from**使用在生成器的定义过程中, 与**yield**对比, `yield from subgen()`这种形式, 可以从**子生成器**中获取结果, 更准确的说是子生成器`subgen`会获得控制权, 并且把产出的值**直接传给**的调用方, 而不再经过生成器, 这种形式:

- 方便了生成器的嵌套使用
- 调用方可以直接调用子生成器

在子生成器执行时, 生成器`gen`会阻塞, 等待子生成器`subgen`交换控制权.

事实上, yield from会调用紧跟对象的`__iter__`方法, 因此对于任何可迭代的对象, 如`str`, `list`等都可以配合`yield from`使用, 代替繁琐的循环:

```python
def gen():
    yield from 'AB'
    yield from range(1, 3)

list(gen())
```

输出为:

```
['A', 'B', '1', '2']
```

#### 执行过程

对主函数直接调用的生成器, 以及上文中的子生成器, 正式定义为:

- **调用方**: 调用委派生成器的客户端代码

- **委派生成器**: 包含yield from表达式的生成器函数
- **子生成器**: 从yield from部分获取的生成器

三者是一种层级递进关系.

**yield from**的主要功能是打开**双向通道**, 把最外层的调用方与最内层的子生成器连接起来, 使两者可以**直接**发送和产出值, 还可以直接传入异常, 而不用在中间的协程添加异常处理的代码.

委派生成器在yield from表达式处暂停时, 调用方可以直接把数据发给子生成器, 子生成器再把产出的值发送给调用方.

子生成器返回之后, 解释器会抛出StopIteration异常, 并把返回值附加到异常对象上, 委派生成器恢复.

从一个例子中感受`yield from`的执行过程.

```python
from collections import namedtuple

Result = namedtuple('Result', 'count average')# 子生成器

# 这个例子和上边示例中的 averager 协程一样，只不过这里是作为字生成器使用

def averager():
    total = 0.0
    count = 0
    average = None
    while True:        # main 函数发送数据到这里 
        term = yield
        if term is None: # 终止条件
            break
        total += term
        count += 1
        average = total/count
    return Result(count, average) # 返回的Result 会成为grouper函数中yield from表达式的值


# 委派生成器
def grouper(results, key):
    # 这个循环每次都会新建一个averager 实例，每个实例都是作为协程使用的生成器对象
    while True:        # grouper 发送的每个值都会经由yield from 处理，通过管道传给averager 实例。grouper会在yield from表达式处暂停，等待averager实例处理客户端发来的值。averager实例运行完毕后，返回的值绑定到results[key] 上。while 循环会不断创建averager实例，处理更多的值。
        results[key] = yield from averager()
    
# 调用方
def main(data):
    results = {}
    for key, values in data.items():
        # group 是调用grouper函数得到的生成器对象，传给grouper 函数的第一个参数是results，用于收集结果；第二个是某个键
        group = grouper(results, key)
        next(group)        
        for value in values:            # 把各个value传给grouper 传入的值最终到达averager函数中；
            # grouper并不知道传入的是什么，同时grouper实例在yield from处暂停
            group.send(value)        # 把None传入groupper，传入的值最终到达averager函数中，导致当前实例终止。然后继续创建下一个实例。
        # 如果没有group.send(None)，那么averager子生成器永远不会终止，委派生成器也永远不会在此激活，也就不会为result[key]赋值
        group.send(None)
    report(results)
        
# 输出报告
def report(results):
    for key, result in sorted(results.items()):
        group, unit = key.split(';')
        print('{:2} {:5} averaging {:.2f}{}'.format(result.count, group, result.average, unit))

data = {
    'girls;kg':[40, 41, 42, 43, 44, 54],
    'girls;m': [1.5, 1.6, 1.8, 1.5, 1.45, 1.6],    
    'boys;kg':[50, 51, 62, 53, 54, 54],    
    'boys;m': [1.6, 1.8, 1.8, 1.7, 1.55, 1.6],
}

if __name__ == '__main__':
    main(data)
```

执行结果为:

```
6 boys  averaging 54.00kg
6 boys  averaging 1.68m
6 girls averaging 44.00kg
6 girls averaging 1.58m
```

上面的代码展示了yield from的基础用法, **委派生成器**相当于管道, 所以可以把任意数量的委派生成器连接在一起. 而且委派生成器中的子生成器内部也可以调用生成器, 从而加深了调用关系, 只需要最终以一个只是用yield表达式的生成器结束.

#### yield from行为总结

- 子生成器产出的值都直接传给委派生成器的调用方
- 使用`send`方法发给委派生成器的值都直接传给子生成器, 如果发送的值是None, 那么会调用子生成器的 **next**()方法; 如果发送的值不是None, 那么会调用子生成器的`send`方法.
  - 如果调用的方法抛出StopIteration异常, 那么委派生成器恢复运行
  - 任何其他异常都会向上冒泡, 传给委派生成器
- 生成器退出时, 委派生成器/子生成器中的**return表达式**`expr`会触发`StopIteration(expr)`, 将异常抛出
- `yield from`表达式的值是子生成器终止时传给StopIteration异常的第一个参数
- 传入委派生成器的异常, 除了**GeneratorExit**之外都传给子生成器的`throw`方法. 如果调用`throw`方法时抛出StopIteration异常, 委派生成器恢复运行. StopIteration之外的异常会向上冒泡, 传给委派生成器
- 如果把GeneratorExit异常传入委派生成器, 或者在委派生成器上调用`close`方法, 那么在子生成器上调用`close`方法(如果子生成器有的话).
  - 如果调用`close`方法导致异常抛出, 那么异常会向上冒泡, 传给委派生成器
  - 否则, 委派生成器抛出GeneratorExit异常

## 参考资料

[python协程2：yield from 分析](https://mp.weixin.qq.com/s/AXaD7vhMYBJdSgxx7Gr3_Q)

[python协程3：用仿真实验学习协程](https://mp.weixin.qq.com/s/C-TODoU5vqwd9qjamTq7Rw)

