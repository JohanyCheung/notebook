## 参考资料

[Hadoop 新 MapReduce 框架 Yarn 详解](https://www.cnblogs.com/gw811/p/4077315.html)

[Hadoop-Yarn-框架原理及运作机制（原理篇）](https://www.cnblogs.com/chushiyaoyue/p/5784871.html)

[初步掌握Yarn的架构及原理](https://www.cnblogs.com/codeOfLife/p/5492740.html)

## 基本架构

由于原来的MapReduce架构中的JobTracker结点承担了太多的任务, 新框架中将其功能进行了拆分. 拆分成了两个独立的服务:

- **ResourceManager**: 全局的**资源管理器**, 专门负责资源管理和调度.
- **ApplicationMaster**: 负责**每个应用**的资源管理, 任务调度, 容错等工作, 每个应用程序对应一个ApplicationMaster

整体的架构如下图:

![](http://www.ibm.com/developerworks/cn/opensource/os-cn-hadoop-yarn/images/image002.jpg)

详细来说ResourceManager, ApplicationMaster, NodeManager三者的作用.

#### ResourceManager

ResourceManager控制整个集群计算资源的分配. 将各部分资源(CPU, 内存, 带宽等)安排给**NodeManager**, 并与NodeManager一起启动和监视ApplicationMaster应用程序. 具体来说有以下作用:

- 处理**客户端**(Client)请求
- 启动和监控每个job所属的**ApplicationMaster**
- 监控NodeManager
- 集群资源的分配与调度

#### ApplicationMaster

单个作业的资源管理和任务监控. 管理协调ResourceManager分配给这个应用的资源, 并通过NodeManager监视资源容器的执行和资源的使用. 总的来说, ApplicationMaster有以下的作用:

- 负责数据的切分
- 为应用程序, 向ResourceManager申请资源(容器), 并分配给应用的子任务
- 启动任务, 对任务进行监控和容错

#### NodeManager

单个节点的资源管理和监控. 每个结点将其资源(CPU, 内存, 带宽等)划分成**Container**, 是YARN架构中的资源单位. 因此NodeManager管理着这个结点上的所有Container. NodeManager作用有:

- 负责Container状态的维护(处理来自ResourceManager和ApplicationMaster的命令)
- 向ResourceManager保持心跳

YARN会为每个任务分配一个Container, 且该任务只能使用该Container中描述的资源.

## 调度过程

![](https://7n.w3cschool.cn/attachments/image/20170808/1502172265232242.jpg)

当用户提交了一个应用之后, 整个过程如下:

- **Job submission**(作业提交)

  - **Client**向**ResourceManager**提交一个**应用/作业**, 从ResourceManager获取一个**Application ID**(应用ID/作业ID).
  - **Client**计算得到输入分片, 将作业资源(job jar, 配置文件, 分片信息)拷贝到HDFS
  - **Client**再向**ResourceManager**提交作业
- **Job initialization**(作业初始化)
  - **ResourceManager**将作业递交给**Scheduler**(调度器), Scheduler为作业分配**第一个Container**, ResourceManager同时于这个Container所在结点的**NodeManager**通信, NodeManager在这个Container中加载这个Job(作业/应用)的**ApplicationMaster**. ApplicationMaster再向ResourceManager注册. 因此就可以通过ResourceManager查看这个Job的运行状态.
  - 这个被创建的ApplicationMaster被交给所处结点的**NodeManager**管理监控.
  - ApplicationMaster获取这个Job的分片, 为每一个分片对应创建一个**Map Task**, 或**Reduce Task**
- **Task assignment**(任务分配)
  - ApplicationMaster向ResourceManager申请资源(Container), 根据输入分片所在的结点, 根据**data locality**, 分配给输入数据分片附近的结点的Container.
- **Task execution**(任务运行)
  - ApplicationMaster根据ResourceManager分配Container的情况, 与对应的NodeManager的通信, 启动这些Container, 读取每个任务(Task)所需的数据(job jar, 配置文件等), 然后执行该任务.
- **Progress and status update**(进度和状态更新)
  - Task定时将任务的进度和状态报告给ApplicationMaster, Client定时向ApplicationMaster获取整个任务的进度和状态.
- **Job completion**(作业完成)
  - Client定时检查整个作业是否完成, 作业完成后, 清空临时文件, 目录等
  - ApplicationMaster向ResourceManager注销, 并关闭自己

