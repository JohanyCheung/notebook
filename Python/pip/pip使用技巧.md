## 安装技巧

### 下载包但不安装

```bash
pip install <包名> -d <目录>
```

### 指定安装源

`-i`参数指定源头, 例如使用常用的阿里源

```bash
pip install <包名> -i https://mirrors.aliyun.com/pypi/simple
```



## 查询

### 查询可升级的包

```bash
pip list -o
```

