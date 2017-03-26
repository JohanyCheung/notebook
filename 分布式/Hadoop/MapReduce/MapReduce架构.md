## 参考资料

[Annotated Hadoop: 第二节 MapReduce框架结构](https://www.cnblogs.com/shipengzhi/articles/2487429.html)

## 总概

MapReduce是一个用于大规模数据处理的**分布式计算模型**, 是一个用于处理和生成大规模数据集的相关的实现. 用户定义一个map函数来处理一个key/value对以生成一批中间的key/value对, 再定义一个reduce函数将所有这些中间的有着相同key的values合并起来, 很多现实世界中的任务都可用这个模型来表达.

## 总体结构

#### JobTracker和TaskTracker

- **JobTracker**运行在**master**结点
  - 是MapReduce框架的中心
  - 需要管理哪些程序应该跑在哪些机器上
  - 需要与集群中的机器定时通信
  - 需要管理所有job失败后重启的操作
- **TaskTracker**运行在**slave**结点
  - 是MapReduce框架集群中每台机器都有的一个部分
  - 主要是监视自己所在机器的资源情况
  - TaskTracker同时监视当前机器的task的运行情况, 将这些信息通过**heartbeat**发送给JobTracker

用户向JobTracker提交**job**, 提交的job由JobTracker进行调度, 调度job的**子任务task**运行在slave结点的TaskTracker上, 并监控它们, 如果有失败的task就重启运行它们.

MapReduce框架的主要组成和工作流程如下图:

![](http://www.cppblog.com/images/cppblog_com/javenstudio/4165/o_hadoop-mapred.jpg)

## 组件

#### JobClient

用户在**用户端**通过**JobClient**类向JobTracker提交job. 具体的提交方法为:

- JobClient类将每一个提交的job的**应用程序**和**配置参数**Configuration打包成jar文件存储在**HDFS**中
- JobClient将这个jar文件的HDFS路径提交给JobTracker

#### Mapper和Reducer

用户编写的**应用程序**最基本的组成部分. 在一些应用中还可以包括**Combiner**类, 它实际也是Reducer的实现.

![](https://7n.w3cschool.cn/attachments/image/wk/hadoop/mapreduce-process-overview.png)

## 运行过程

分成用户提交job的任务分法**调度**过程, 以及环境之间相互**通信**的过程.

#### 调度

- **用户端程序**JobClient向JobTracker提交了一个job
  - 在一个job中, 包含了程序中的Mapper和Reducer, 以及job的配置参数**JobConf**
  - **提交方法**: JobClient将程序和配置参数打包成jar文件存储在HDFS上, 并把这个文件对应的文件路径提交给JobTracker
- JobTracker接收到JobClient提交的job后, 会创建一个**JobInProgress**对象, 通过这个对象跟踪并监督这个job, 目的是管理和调度这个job对应的**子任务**task
  - JobInProgress对象会根据提交的job对应的jar中定义的已分片的输入数据集, 创建对应的一批**TaskInProgress**对象. 这些TaskInProgress分为两类:
    - 与分片输入数据集数量对应的, 用于**监控**和**调度**的**MapTask**
    - 指定数目的, 用于**监控**和**调度**的**ReduceTask**, 默认值为1
- 每创建一个TaskInProgress, 都会启动(Launch)一个**Task**对象(**MapTask**/**ReduceTask**), 序列化这个Task, 写入到计算得到的最优化的**TaskTracker**结点中去
- TaskTracker收到这个序列化的Task, 会创建一个**TaskInProgress**(与JobTracker中的TaskInProgress非同一个类, 但作用类似), 用来**监控**和**调度**该Task
- 通过TaskInProgress管理的**TaskRunner**对象来启动具体的Task进程, 具体来说:
  - TaskRunner自动装载job对应的jar(从HDFS系统中), 设置好环境变量, 启动一个独立的java子进程来执行Task(**MapTask**/**ReduceTask**)
  - **程序**中的Mapper和Combiner由MapTask调用执行, Reducer由ReduceTask调用执行, **两者不一定在同一个TaskTracker结点中**

#### 通信

- TaskTracker监视当前机器的Task运行状况
- TaskTracker通过heartbeat把Task运行情况的信息发送给JobTracker
- JobTracker搜集这些信息, 给新提交的job分配运行所需的机器, 即TaskTracker

## 缺点

- JobTracker是MapReduce的**集中处理点**, 存在**单点故障**
- JobTracker完成了太多的任务, 造成了过多的资源消耗. 当job非常多的时候, 会造成很大的内存开销, 也增加了JobTracker fail的风险
- TaskTracker以**map/reduce task**的数目作为资源的表示过于简单, 没有考虑到CPU/内存的占用情况, 如果两个内存消耗大的task被调度在了一起, 很容易出现**OOM**
- TaskTracker把资源强制划分为**map task slot**和**reduce task slot**, 当系统中只有map task或reduce task的时候, 会造成资源的浪费

