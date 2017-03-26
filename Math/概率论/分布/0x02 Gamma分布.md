## Gamma函数

Gamma函数是一个是实数域的函数:

$$\Gamma(x)=\int_0^{\infty}t^{x-1}e^{-t}dt$$

形如:

![](http://cos.name/wp-content/uploads/2013/01/gamma-func.png)

具有性质:

- **递归性质**: $$\Gamma(x+1) = x \Gamma(x)$$
- $$\Gamma(x)$$函数可以当成是阶乘在**实数集**上的延拓, 即在实数集上的**阶乘函数**: $$\Gamma(n) = (n-1)!$$

## Gamma分布

#### Gamma分布的概率密度

Gamma函数在概率统计中与众多的统计分布有关, 包括常见的统计学三大分布($$t$$分布, $$\chi^2$$分布, $$F$$分布), Beta分布, Dirichlet分布的**密度公式**中都有Gamma函数的身影.

当然发生最直接联系的概率分布是直接由Gamma函数变换得到的**Gamma分布**.

对Gamma函数的定义做一个变形, 可以得到:

$$\int_0^{\infty} \frac{x^{\alpha-1}e^{-x}}{\Gamma(\alpha)}dx = 1$$

积分值为1, 所以积分中的函数就是一个**概率密度函数**, 这就是Gamma分布的概率密度函数:

$$Gamma(x|\alpha) = \frac{x^{\alpha-1}e^{-x}}{\Gamma(\alpha)}$$

做一个变换$$x=\beta t$$, 就得到Gamma分布的更一般的形式:

$$Gamma(t|\alpha, \beta) = \frac{\beta^\alpha t^{\alpha-1}e^{-\beta t}}{\Gamma(\alpha)}$$

所以Gamma分布有两个参数:

- $$\alpha$$称为**shape parameter**, 主要决定了分布曲线的形状
- $$\beta$$称为**rate parameter**或**inverse scale parameter**, 主要决定曲线有多陡

不同参数下Gamma分布的概率密度图如下所示:

![](http://cos.name/wp-content/uploads/2013/01/gamma-distribution.png)

## Gamma分布的迷人之处

Gamma分布与Gamma函数一样, 在概率统计领域也是一个万人迷, 众多统计分布和它有密切关系.

- Gamma分布作为**先验分布**很强大, 在**贝叶斯统计分析**中被广泛的用作**其它分布**的先验
- **指数分布**和**$$\chi^2$$分布**都是特殊的Gamma分布
- 与以下分布有着**共轭关系**: **指数分布**, **泊松(Poission)分布**, **正态分布**, **对数正态分布**

#### 与Poission的一致性

参数为$$\lambda$$的Poisson分布, 概率写为:

$$Poisson(X=k|\lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Gamma分布的密度中取$$\alpha = k+1$$, $$\beta=1$$得到:

$$Gamma(x|\alpha=k+1) = \frac{x^ke^{-x}}{\Gamma(k+1)}= \frac{x^k e^{-x}}{k!}$$

所以这两个分布数学形式上是一致的, 只是Poisson分布是**离散**的, Gamma分布是**连续**的

## 参考资料

[[LDA数学八卦-1]神奇的Gamma函数](http://www.flickering.cn/数学之美/2014/06/[lda数学八卦]神奇的gamma函数/)

