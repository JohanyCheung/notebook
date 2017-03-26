## 时间类型格式

使用[Series.dt](http://pandas.pydata.org/pandas-docs/stable/reference/series.html#api-series-dt)访问器, 需要先将当前Series转换为**datetimelike**类型的格式, 在pandas中指的就是:

- Datetime
- Timedelta
- Period

## 格式转化

### Datetime

- 可以在IO时直接将字符串表示的时间格式, 通过一定的设置, 转换成**Datetime**的类型.

    例如如下的数据格式:

    ```python
    data = pd.read_csv(file_path)
    data.head()
    ```

    ```
            Month	#Passengers
    0	1949-01	112
    1	1949-02	118
    2	1949-03	132
    3	1949-04	129
    4	1949-05	121
    ```

    其中的`Month`列就符合`%Y-%m`的格式, 因此读取方法改为:

    ```python
    data = pd.read_csv(file_path, parse_dates=["Month"], date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m"))
    data.head()
    ```

    ```
            Month	#Passengers
    0	1949-01-01	112
    1	1949-02-01	118
    2	1949-03-01	132
    3	1949-04-01	129
    4	1949-05-01	121
    ```

    或者直接使用自动推断的方法, 更简单省事, 对于一般的格式都能很好的完成:

    ```python
    data = pd.read_csv(file_path, parse_dates=["Month"], infer_datetime_format=True)
    ```

- 另外也可以使用[pandas.to_datetime](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html#pandas.to_datetime)函数, 将指定的一列转为Datetime格式, 方法更灵活.

    ```python
    pandas.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, box=True, format=None, exact=True, unit=None, infer_datetime_format=False, origin='unix', cache=False)
    ```

    其中比较重要的参数有:

    - arg: integer, float, string, datetime, list, tuple, 1-d array, Series
      - 要转换的数据, 注意支持多种格式的数据
    - errors: {‘ignore’, ‘raise’, ‘coerce’}, default ‘raise’
      - 遇到无法转换的脏数据如何处理
      - `coerce`方法将会对应返回**NaT**
    - format: string, default None
    - infer_datetime_format: boolean, default False
      - 自动推断格式, 不需要给出`format`

    转换例子如下:

    ```python
    >>> df = pd.DataFrame({'year': [2015, 2016],
                       'month': [2, 3],
                       'day': [4, 5]})
    >>> pd.to_datetime(df)
    0   2015-02-04
    1   2016-03-05
    dtype: datetime64[ns]
    ```

    ```
    >>> pd.to_datetime('13000101', format='%Y%m%d', errors='ignore')
    datetime.datetime(1300, 1, 1, 0, 0)
    >>> pd.to_datetime('13000101', format='%Y%m%d', errors='coerce')
    NaT
    ```

    ```python
    >>> pd.to_datetime(1490195805, unit='s')
    Timestamp('2017-03-22 15:16:45')
    >>> pd.to_datetime(1490195805433502912, unit='ns')
    Timestamp('2017-03-22 15:16:45.433502912')
    ```

    ```python
    >>> pd.to_datetime([1, 2, 3], unit='D',
                    origin=pd.Timestamp('1960-01-01'))
    0    1960-01-02
    1    1960-01-03
    2    1960-01-04
    ```

### Timedelta

使用[pandas.to_timedelta](http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html#pandas.to_timedelta)函数进行转化.

```python
pandas.to_timedelta(arg, unit='ns', box=True, errors='raise')
```

```python
>>> pd.to_timedelta('1 days 06:05:01.00003')
Timedelta('1 days 06:05:01.000030')
>>> pd.to_timedelta('15.5us')
Timedelta('0 days 00:00:00.000015')
```

```python
>>> pd.to_timedelta(['1 days 06:05:01.00003', '15.5us', 'nan'])
TimedeltaIndex(['1 days 06:05:01.000030', '0 days 00:00:00.000015', NaT],
               dtype='timedelta64[ns]', freq=None)
```

```python
>>> pd.to_timedelta(np.arange(5), unit='s')
TimedeltaIndex(['00:00:00', '00:00:01', '00:00:02',
                '00:00:03', '00:00:04'],
               dtype='timedelta64[ns]', freq=None)
>>> pd.to_timedelta(np.arange(5), unit='d')
TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
               dtype='timedelta64[ns]', freq=None)
```

```python
>>> pd.to_timedelta(np.arange(5), box=False)
array([0, 1, 2, 3, 4], dtype='timedelta64[ns]')
```

## dt中的重要方法

- Series.dt.date
  - 返回一个元素格式为python中的**datetime.date**对象的numpy array
  - 之后对于其中的每个元素就可以使用`datetime`中的`strftime`等方法进行继续的操作
- Series.dt.time
  - - 返回一个元素格式为python中的**datetime.time**对象的numpy array
- Series.dt.year
- Series.dt.month
- Series.dt.day
- Series.dt.hour
- Series.dt.minute
- Series.dt.second
- Series.dt.microsecond
- Series.dt.nanosecond
- Series.dt.week / Series.dt.weekofyear
- Series.dt.weekday / Series.dt.dayofweek
  - Monday=0, Sunday=6
- Series.dt.quarter
