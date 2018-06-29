## Dirichlet分布

#### 引入

均匀生成20个[0,1]之间的随机数, 同时第7大和第13大的数是什么?

数学表达:

- $$X_1,X_2,\cdots,X_n {\stackrel{\mathrm{iid}} {\sim}}Uniform(0,1)$$, 排序后对应的顺序统计量$$X_{(1)},X_{(2)}，\cdots, X_{(n)}$$
- 问$$(X_{(k_1)}, X_{(k_1+k_2)})$$的联合分布是什么

#### 推导得到Dirichlet分布

![](http://cos.name/wp-content/uploads/2013/01/dirichlet-game.png)

取$$x_3$$满足$$x_1+x_2+x_3 = 1$$, $$(X_{(k_1)}, X_{(k_1+k_2)})$$如下:

$$\begin{align*}  & P\Bigl(X_{(k_1)}\in(x_1,x_1+\Delta x),X_{(k_1+k_2)}\in(x_2,x_2+\Delta x)\Bigr) \\  & = n(n-1)\binom{n-2}{k_1-1,k_2-1}x_1^{k_1-1}x_2^{k_2-1}x_3^{n-k_1-k_2}(\Delta x)^2 \\  & = \frac{n!}{(k_1-1)!(k_2-1)!(n-k_1-k_2)!}x_1^{k_1-1}x_2^{k_2-1}x_3^{n-k_1-k_2}(\Delta x)^2  \end{align*}$$

于是我们得到$$(X_{(k_1)}, X_{(k_1+k_2)})$$的联合分布是:

$$\begin{align*}  f(x_1,x_2,x_3) & = \frac{n!}{(k_1-1)!(k_2-1)!(n-k_1-k_2)!}x_1^{k_1-1}x_2^{k_2-1}x_3^{n-k_1-k_2} \\  & = \frac{\Gamma(n+1)}{\Gamma(k_1)\Gamma(k_2)\Gamma(n-k_1-k_2+1)}x_1^{k_1-1}x_2^{k_2-1}x_3^{n-k_1-k_2}  \end{align*}$$

上面这个分布其实就是3维形式的**Dirichlet分布**$$Dir(x_1,x_2,x_3|k_1,k_2,n-k_1-k_2+1)$$, 令$$\alpha_1=k_1,\alpha_2=k_2,\alpha_3=n-k_1-k_2+1$$, 分布密度可以写为:

$$\begin{equation}  \displaystyle f(x_1,x_2,x_3) = \frac{\Gamma(\alpha_1 + \alpha_2 + \alpha_3)}  {\Gamma(\alpha_1)\Gamma(\alpha_2)\Gamma(\alpha_3)}x_1^{\alpha_1-1}x_2^{\alpha_2-1}x_3^{\alpha_3-1}  \end{equation}$$

这就是一般形式的3维**Dirichlet分布**, 从形式上我们也能看出, **Dirichlet分布是Beta分布在高维度上的推广**, 和Beta分布一样也是一个百变星君, 密度函数可以展现出多种形态:

![](http://cos.name/wp-content/uploads/2013/01/dirichlet-distribution.png)

一般形式的**Dirichlet分布**定义如下:

$$\begin{equation}  \displaystyle Dir(\overrightarrow{p}|\overrightarrow{\alpha}) =  \displaystyle \frac{\Gamma(\sum_{k=1}^K\alpha_k)}  {\prod_{k=1}^K\Gamma(\alpha_k)} \prod_{k=1}^K p_k^{\alpha_k-1}  \end{equation}$$

## Dirichlet-Multinomial共轭

#### 引入

调整一下游戏, 从魔盒中生成$$m$$个随机数$$Y_1,Y_2,\cdots,Y_m {\stackrel{\mathrm{iid}}{\sim}}Uniform(0,1)$$, 且知道$$Y_i$$和$$(X_{(k_1)}, X_{(k_1+k_2)})$$相比谁大谁小, 再求上面的联合分布(后验分布).

数学表示如下:

- $$X_1,X_2,\cdots,X_n {\stackrel{\mathrm{iid}} {\sim}}Uniform(0,1)$$, 排序后对应的顺序统计量$$X_{(1)},X_{(2)}，\cdots, X_{(n)}$$
- 令$$p_1=X_{(k_1)}, p_2=X_{(k_1+k_2)},p_3 = 1-p_1-p_2$$, 猜测$$\overrightarrow{p}=(p_1,p_2,p_3)$$
- $$Y_1,Y_2,\cdots,Y_m {\stackrel{\mathrm{iid}}{\sim}}Uniform(0,1)$$, $$Y_i$$中落到$$[0,p_1),[p_1,p_2),[p_2,1]$$三个区间的个数分别为$$m=m_1+m_2+m3$$
- 问后验分布$$P(\overrightarrow{p}|Y_1,Y_2,\cdots,Y_m)$$的分布是什么

#### 推导

记$$\overrightarrow{m}=(m_1,m_2,m_3),\quad \overrightarrow{k}=(k_1,k_2,n-k_1-k_2+1)$$, 由游戏中的信息, 我们可以推理得到$$p_1, p_2$$在$$X_1,X_2,\cdots,X_n,Y_1,Y_2,\cdots,Y_m{\stackrel{\mathrm{iid}}{\sim}} Uniform(0,1)$$这$$m+n$$个数中分别成为了第$$k_1+m_1, k_2+m_2$$大的数, 于是后验分布$$P(\overrightarrow{p}|Y_1,Y_2,\cdots,Y_m)$$应该是:

$$Dir(\overrightarrow{p}|k_1+m_1,k_1+m_2,n-k_1-k_2+1+m_3)$$

即$$Dir(\overrightarrow{p}|\overrightarrow{k}+\overrightarrow{m})$$. 按照贝叶斯推理的逻辑, 同样可以把以上过程整理如下:

- 要猜测参数$$\overrightarrow{p}=(p_1,p_2,p_3)$$, 其先验分布为$$Dir(\overrightarrow{p}|\overrightarrow{k})$$
- 数据$$Y_i$$落到$$[0,p_1),[p_1,p_2),[p_2,1]$$三个区间的个数分别为$$m_1,m_2,m_3$$, 服从**多项分布**$$Mult(\overrightarrow{m}|\overrightarrow{p})$$
- 在给定了来自数据提供的知识$$\overrightarrow{m}$$, $$\overrightarrow{p}$$的后验分布变为$$Dir(\overrightarrow{p}|\overrightarrow{k}+\overrightarrow{m})$$

以上贝叶斯分析过程的简单直观的表述就是:

$$Dir(\overrightarrow{p}|\overrightarrow{k}) + MultCount(\overrightarrow{m}) = Dir(\overrightarrow{p}|\overrightarrow{k}+\overrightarrow{m})$$

令$$\overrightarrow{\alpha}=\overrightarrow{k}$$, 把$$\overrightarrow{\alpha}$$从整数集合延拓到实数集合, 更一般的可以证明有如下关系:

$$\begin{equation}  Dir(\overrightarrow{p}|\overrightarrow{\alpha}) + MultCount(\overrightarrow{m})  = Dir(p|\overrightarrow{\alpha}+\overrightarrow{m})  \end{equation}$$

以上式子实际上描述的就是**Dirichlet-Multinomial共轭**, 从以上过程可以看到, Dirichlet 分布中的参数$$\overrightarrow{\alpha}$$可以理解为物理计数, 类似于Beta分布.

对于给定的$$\overrightarrow{p}$$和$$N$$, 多项分布定义为:

$$\begin{equation}  \displaystyle Mult(\overrightarrow{n} |\overrightarrow{p},N) = \binom{N}{\overrightarrow{n}}\prod_{k=1}^K p_k^{n_k}  \end{equation}$$

$$Mult(\overrightarrow{n} |\overrightarrow{p},N)$$和$$Dir(\overrightarrow{p}|\overrightarrow{\alpha})$$这两个分布是共轭关系.

## Dirichlet分布性质

类似于Beta分布, 如果$$\overrightarrow{p} \sim Dir(\overrightarrow{t}|\overrightarrow{\alpha})$$, Dirichlet分布的期望/均值为:

$$\begin{equation}  E(\overrightarrow{p}) = \Bigl(\frac{\alpha_1}{\sum_{i=1}^K\alpha_i},\frac{\alpha_2}{\sum_{i=1}^K\alpha_i},\cdots, \frac{\alpha_K}{\sum_{i=1}^K\alpha_i} \Bigr)  \end{equation}$$

这个结论很重要, 例如在LDA数学推导中就需要使用到这个结论.

