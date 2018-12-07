## AR模型

**AR(Autoregressive, 自回归)模型**, 描述的是当前值与**历史值**之间的关系:

$$X_t=c+\sum\limits_{i=1}^p\varphi_iX_{t-i}+\varepsilon_i$$

其中$$\varepsilon_i$$是**AR模型**拟合后的剩余部分(即其他因素造成的), 称为**自回归部分的随机误差**. 有如下的特征:

- 所有时间点$$t$$对应的$$\varepsilon_t$$都是一个**期望值为0**的正态分布, 但每个时间的对应正态分布的方差$$\sigma^2_t$$不同
- 不同时间点之间的$$\varepsilon_t$$, 相互之间没有任何联系

## MA模型

**MA(Moving Average, 移动平均)模型**, 公式如下:

$$X_t=\mu+\varepsilon_t+\sum\limits_{i=1}^{q}\theta_i\varepsilon_{t-i}$$

可以看出, MA模型描述的是自回归模型在每个时间点上的误差的累计模型. 之所以成为移动平均, 首先是每个当前值只考虑若干个之前的误差, 体现了移动; 另外, 由于每个时间点上自回归模型的误差如上面所说, 是独立的, 因此可以代表对应时间的序列, 对它们使用带权值的加和体现了平均的性质.

## ARIMA模型

**ARIMA(Autoregressive Integrated Moving Average model)模型**就是将上面两个模型进行综合, 并在使用**ARMA**模型拟合之前, 进行**差分**(Integrated)操作.

$$X_t = \varphi_0 + \sum_{i=1}^p \varphi_i X_{t-i} + \varepsilon_t - \sum_{j=1}^q \theta_j \varepsilon_{t-j}$$

上式中的$$X_t$$是经过$$d$$阶差分之后的序列, 这点需要注意, 差分并没有在公式中体现出来.

因此ARIMA的需要调整的参数为$$(p, d, q)$$, 对应于**AR模型参数**, **差分阶数**, **MA模型参数**.

另外需要注意的是ARIMA模型是对**平稳时间序列**建模的工具, 因此需要通过差分, 将非平稳序列转换为平稳序列.

## ARIMA模型参数的确定

- 通过**ACF(Autocorrelation Function, 自相关函数)**和**PACF(Partial Autocorrelation Function, 偏自相关函数)**, 做出两张图, 根据ACF图和PACF的表现, 主观的确定

- **评价函数衡量**, 如使用**AIC赤池信息量**和**BIC贝叶斯信息量**来衡量当前参数对应的ARIMA模型的拟合效果, 综合选取拟合效果更好的参数作为最优的参数组合. 类似于机器学习中的调参过程.

## Python中的ARIMA模型

```python
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
```

ARIMA模型的初始化参数有:

```python
ARIMA(endog, order, exog=None, dates=None, freq=None, missing='none')
```

- endog: 现有的时间序列
- order: 模型参数, 以(p, d, q)组成的三元`tuple`形式传递
- exog: 外部变量, 作为时间变量之外的参数为拟合模型所使用. 如果使用外部变量, 模型就变为了**ARIMAX**, X指的是外部变量

ARIMA模型通过`.fit()`方法进行拟合训练. `.fit()`函数接受的参数为:

```python
fit(self, start_params=None, trend='c', method="css-mle",
            transparams=True, solver='lbfgs', maxiter=500, full_output=1,
            disp=5, callback=None, start_ar_lags=None, **kwargs)
```

- disp: 打印轮数, 如果`disp<0`, 则不打印信息

---

`.fit()`方法返回一个`statsmodels.tsa.arima.ARIMAResults`对象, 作为拟合的结果. 该对象包含了多种信息:

- `ARIMAResults.fittedvalues`: 拟合历史的结果

---

以及通过`.predict()`方法预测时间序列的未来:

```python
predict(self, params, start=None, end=None, exog=None, typ='linear', dynamic=False)
```

