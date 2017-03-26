## Beta分布
#### 引入
有一个魔盒，上面有一个按钮，你每按一下按钮，就均匀的输出一个[0,1]之间的随机数，我现在按10下，我手上有10个数，你猜第7大的数是什么，偏离不超过0.01就算对.

数学抽象如下:

- $$X_1,X_2,\cdots,X_n {\stackrel{\mathrm{iid}}{\sim}} Uniform(0,1)$$
- 把这$$n$$个随机变量排序后得到顺序统计量$$X_{(1)},X_{(2)}，\cdots, X_{(n)}$$
- 然后求解$$X_{(k)}$$的分布是什么

==**很重要的题外话**: 在概率统计学中, 几乎所有重要的概率分布都可以从**均匀分布**$$Uniform(0,1)$$中生成出来. 尤其是在统计模拟中, 所有统计分布的随机样本都是通过均匀分布产生的.==

对于上面的游戏而言, $$n=10,k=7$$, 如果我们能求出$$X_{(7)}$$的分布的概率密度, 用概率密度的极值点去做猜测就是最好的策略.

#### 推导得到Beta分布

尝试计算$$X_{(k)}$$落在一个区间$$[x, x+\Delta x]$$的概率, 即求:

$$P( x \le X_{(k)} \le x+\Delta x) = ?$$

把$$ [0,1]$$分成三段: $$[0,x), [x,x+\Delta x], (x+\Delta x,1]$$. 先考虑简单的情形: 假设$$n$$个数中只有**一个**落在了区间$$[x, x+\Delta x]$$内, 则$$[0,x)$$中应该有$$k-1$$个数, $$(x,1]$$中应该有$$n-k$$个数. 构造一个符合上述要求的事件$$E$$:

$$\begin{align*}  E = \{ & X_1 \in [x, x+\Delta x], \\  & X_i \in [0,x)\quad (i=2,\cdots,k), \\  & X_j \in (x+\Delta x,1] \quad (j=k+1,\cdots,n)\}  \end{align*}$$

![](http://cos.name/wp-content/uploads/2013/01/beta-game-1.png)

则有:

$$\begin{align*}  P(E) & = \prod_{i=1}^nP(X_i) \\  & = x^{k-1}(1-x-\Delta x)^{n-k}\Delta x \\  & = x^{k-1}(1-x)^{n-k}\Delta x + o(\Delta x)  \end{align*}$$

$$o(\Delta x)$$表示$$\Delta x$$的**高阶无穷小**.

继续考虑稍微复杂一点情形, 假设$$n$$个数中有两个数落在了区间$$[x, x+\Delta x]$$中, 对应的事件为:

$$\begin{align*}  E’ = \{ & X_1,X_2\in [x, x+\Delta x], \\  & X_i \in [0,x) \quad (i=3,\cdots,k), \\  & X_j \in (x+\Delta x,1] \quad (j=k+1,\cdots,n)\}  \end{align*}$$

![](http://cos.name/wp-content/uploads/2013/01/beta-game-2.png)

则有:

$$P(E’) = x^{k-2}(1-x-\Delta x)^{n-k}(\Delta x)^2 = o(\Delta x)$$

因此只要落在$$[x, x+\Delta x]$$内的数字超过一个, 则对应的事件的概率就是$$o(\Delta x)$$. 于是有:

$$\begin{align*}  & P( x \le X_{(k)} \le x+\Delta x) \\  & = n\binom{n-1}{k-1}P(E) + o(\Delta x) \\  & = n\binom{n-1}{k-1}x^{k-1}(1-x)^{n-k}\Delta x + o(\Delta x)  \end{align*}$$

其中$$P(E)$$前面的系数产生的原因为: $$n$$个数中有一个落在$$[x, x+\Delta x]$$区间的有$$n$$种取法, 余下$$n-1$$个数中有$$k-1$$个落在$$[0,x)$$中的有$$\binom{n-1}{k-1}$$种组合, 所以跟事件$$E$$等价的事件一共有$$n\binom{n-1}{k-1}$$个.

根据上式, 可以得到$$X_{(k)}$$概率密度函数为:

$$\begin{align*}  f(x) & = \lim_{\Delta x \rightarrow 0} \frac{P( x \le X_{(k)} \le x+\Delta x)}{\Delta x} \\  & = n\binom{n-1}{k-1}x^{k-1}(1-x)^{n-k} \\  & = \frac{n!}{(k-1)!(n-k)!}x^{k-1}(1-x)^{n-k} \quad x \in [0,1]  \end{align*}$$

利用**Gamma函数**, 可以把$$f(x)$$表达为:

$$f(x) = \frac{\Gamma(n+1)}{\Gamma(k)\Gamma(n-k+1)}x^{k-1}(1-x)^{n-k}$$

取$$\alpha=k, \beta=n-k+1$$, 得到:

$$\begin{equation}  f(x) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1} \tag{1} \end{equation}$$

**这个就是一般意义上的Beta分布**.

## Beta-Binomial共轭

#### 引入

对于上面的游戏, 再给5个[0,1]之间的随机数, 告诉这5个数中的每一个和第7大的数相比, 谁大谁小, 然后继续猜第7大的数是多少.

数学形式为:

- $$X_1,X_2,\cdots,X_n {\stackrel{\mathrm{iid}}{\sim}}Uniform(0,1)$$, 对应的顺序统计量是$$X_{(1)},X_{(2)}，\cdots, X_{(n)}$$, 我们要猜测$$p=X_{(k)}$$
- $$Y_1,Y_2,\cdots,Y_m {\stackrel{\mathrm{iid}}{\sim}}Uniform(0,1)$$, $$Y_i$$中有$$m_1$$个比$$p$$小, $$m_2$$个比$$p$$大
- 问$$P(p|Y_1,Y_2,\cdots,Y_m)$$的分布是什么

#### 推导

我们容易推理得到$$p=X_{(k)}$$在$$X_1,X_2,\cdots,X_n,Y_1,Y_2,\cdots,Y_m {\stackrel{\mathrm{iid}}{\sim}} Uniform(0,1)$$这$$(m+n)$$个独立随机变量中是第$$k+m_1$$大的, 根据上一个小节的推理, 此时$$p=X_{(k)}$$概率密度函数符合Beta分布, 为$$Beta(p|k+m_1,n-k+1+m_2)$$, 得到了问题的答案.

根据**贝叶斯推理**的逻辑, 把以上过程整理如下:

- $$p=X_{(k)}$$是我们要猜测的参数, 推导出$$p$$的分布为$$f(p) = Beta(p|k,n-k+1)$$, 称为$$p$$的先验分布
- 数据$$Y_i$$中有$$m_1$$个比$$p$$小, $$m_2$$个比$$p$$大, $$Y_i$$相当于是做了$$m$$次**贝努利实验**, 所以$$m_1$$服从**二项分布**$$B(m,p)$$
- 在给定了来自数据提供的$$(m_1,m_2)$$的知识后, $$p$$的后验分布变为$$f(p|m_1,m_2)=Beta(p|k+m_1,n-k+1+m_2)$$

贝叶斯参数估计的基本过程是**先验分布 + 数据的知识 = 后验分布**, 以上贝叶斯分析过程的简单直观的表述就是:

$$Beta(p|k,n-k+1) + Count(m_1,m_2) = Beta(p|k+m_1,n-k+1+m_2)$$

其中$$(m_1,m_2)$$对应的是二项分布$$B(m_1+m_2,p)$$的计数.

更一般的, 对于非负实数$$\alpha,\beta$$, 有如下关系:

$$\begin{equation}  Beta(p|\alpha,\beta) + Count(m_1,m_2) = Beta(p|\alpha+m_1,\beta+m_2) \tag{2} \end{equation}$$

这个式子实际上描述的就是**Beta-Binomial共轭**.

==**共轭的意思就是**==: 此处, 数据符合二项分布的时候, 参数的先验分布和后验分布都能保持**Beta分布**的形式. 这种形式不变的好处是, 我们能够**在先验分布中赋予参数很明确的物理意义**, 同时从先验变换到后验过程中从数据中补充的知识也容易有物理解释.

Beta分布中的参数$$\alpha,\beta$$都可以理解为**物理计数**, 这两个参数经常被称为**伪计数**(pseudo-count). 因为我们可以把一个Beta分布$$Beta(p|\alpha,\beta)$$写成下式来理解:

$$Beta(p|1,1) + Count(\alpha-1,\beta-1) = Beta(p|\alpha,\beta)$$

其中$$Beta(p|1,1)$$恰好就是均匀分布$$Uniform(0,1)$$.

## Beta分布性质

![](http://cos.name/wp-content/uploads/2013/01/beta-distribution.png)

把Beta分布的**概率密度**画成图, 会发现它是个百变星君, 它可以是**凹的**, **凸的**, **单调上升的**, **单调下降的**, 可以是**曲线**也可以是**直线**.

而**均匀分布**也是特殊的Beta分布.

由于Beta 分布能够拟合如此之多的形状, 因此它在统计数据**拟合**中被广泛使用.

---

另外, 如果$$p\sim Beta(t|\alpha,\beta)$$, 则有:

$$\begin{align*}  E(p) & = \int_0^1 t*Beta(t|\alpha,\beta)dt \\  & = \int_0^1 t* \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} t^{\alpha-1}(1-t)^{\beta-1}dt \\  & = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} \int_0^1 t^{\alpha}(1-t)^{\beta-1}dt  \end{align*}$$

上式右边的积分对应到概率分布$$Beta(t|\alpha+1,\beta)$$, 对于这个分布我们有:

$$\int_0^1 \frac{\Gamma(\alpha+\beta+1)}{\Gamma(\alpha+1)\Gamma(\beta)} t^{\alpha}(1-t)^{\beta-1}dt = 1$$

带入上式得到:

$$\begin{align}  E(p) & = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}  \cdot \frac{\Gamma(\alpha+1)\Gamma(\beta)}{\Gamma(\alpha+\beta+1)} \notag \\  & = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha+\beta+1)}\frac{\Gamma(\alpha+1)}{\Gamma(\alpha)} \notag \\  & = \frac{\alpha}{\alpha+\beta} \end{align}$$

这说明对于Beta 分布的随机变量, 其均值可以用$$\frac{\alpha}{\alpha+\beta}$$来估计.

这个结论很重要, 例如在LDA数学推导中就需要使用到这个结论.

