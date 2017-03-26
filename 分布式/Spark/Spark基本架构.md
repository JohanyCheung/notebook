## 引入

从全局的框架角度来看, **spark**的位置如下:

![](https://images2015.cnblogs.com/blog/1004194/201608/1004194-20160829161404996-1972748563.png)

spark的上方是向用户开放的各种接口API, 下方是分布式运行的框架, 可以运行在spark本身提供的分布式运行框架上, 即以**独立运行模式(Standalone)**工作, 也可以借助其他的分布式框架运行, 如常见的YARN架构. 除此之外, spark也需要运行在分布式存储架构上, 如HDFS等.

## 概念

先从spark结构的基础概念入手.

#### 系统结点

- **Client**: 用户使用的机器环境
- **Master**
- **Worker**

#### 结点角色

- **Driver**: 创建**SparkContext**, 运行**Application**中用户编写的**main函数**. SparkContext为spark应用程序的运行准备了环境, 同时负责与**ClusterManager**进行通信, 进行资源的申请, 任务的分配和监控等. 通常SparkContext就代表着Driver.
- **ClusterManager**: 本质是**外部服务**, 作用是获取集群中的资源. 因此可以直观地理解YARN中的ResourceManager可以承担此角色. Spark在不同的模式下运行时, 不同的组间承担此角色:
  - Standalone: spark原生的资源管理, 由Master负责资源的分配
  - YARN: Yarn中的ResourceManager
- **WorkerNode**: 集群中任何可以运行Application代码的节点.
  - Standalone: 通过slave文件配置的Worker节点
  - YARN: NoteManager结点

#### 控制流

- **Appliction**: 指用户编写的Spark应用程序, 包含:
  - Driver功能的代码
  - 分布在集群中多个节点上运行的Executor代码
- **Job**: 由Spark中的**Action**动作触发生成一个Job, 因此一个Application中往往会产生多个Job. 每个Job包含多个Task组成的并行计算.
- **Stage**: 每个Job会被拆分成**多组Task**, 成为一个**TaskSet**, 称为一个Stage.
  - Stage的划分和调度是由**DAGScheduler**来负责的
  - Stage分为**Shuffle Map Stage**和**Result Stage**两种, Stage的边界就是**发生shuffle**的地方
- **Task**: 被送到某个Executor上的工作单元, 是运行Application的基本单位. 多个Task组成一个Stage.
  - Task的调度和管理等是由**TaskScheduler**负责

#### SparkContext

SparkContext的整体架构和作用如下图所示:

![](https://images2015.cnblogs.com/blog/1004194/201608/1004194-20160830094200918-1846127221.png)

- **DAG Graph**: 首先将用户编写的spark代码中包含的RDD对象的关系, 翻译成DAG Graph

  ![](https://images2015.cnblogs.com/blog/1004194/201608/1004194-20160830110240699-1379053598.png)

- **DAGScheduler**: DAGScheduler根据**Job**, 将原有的DAG图划分为多个Stage, 构成基于Stage的DAG, 并提交Stage给**TASkScheduler**.

  - 划分Stage的依据是: RDD之间的依赖的关系找出开销最小的调度方法
  - 当碰到Action操作时, 就会催生Job; 每个Job中含有1个或多个Stage, Stage一般在获取外部数据和shuffle之前产生

- **TaskScheduler**: 将**TaskSet**交给**WorkerNode**运行, WorkerNode中的每个**Executor**运行什么Task就是在此处分配的. TaskScheduler还维护着所有Task的运行标签, 重试失败的Task.

  - 当Executor向Driver发生心跳时, TaskScheduler会根据资源剩余情况分配相应的Task

  ![](https://images2015.cnblogs.com/blog/1004194/201608/1004194-20160830110703918-1499305788.png)

## 框架总括

**Spark框架**的组成图如下:

![](https://images2015.cnblogs.com/blog/1004194/201608/1004194-20160829174157699-296881431.png)

整个**执行流程图**如下图:

![](https://images2015.cnblogs.com/blog/1004194/201608/1004194-20160829182313371-1648664691.png)

