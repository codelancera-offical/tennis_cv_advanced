---
type: knowledge_card
tags:
  - python
  - blog
  - conda
date_creation: 2024-07-10 16:54
---
```dataview
TABLE dateformat(this.file.mtime, "yyyy-MM-dd HH:mm:ss") AS "Last Modified Time"
WHERE file.path = this.file.path
```

anaconda的下载并且[[Python开发环境搭建#Anaconda配置|基本配置]]后的base环境已经包含了基本的数据分析和机器学习所需要的库。

但是你需要进行环境管理的话，如下是你会经常用到的命令:

**信息类命令**
- `conda info`：输出conda的基本信息，包括源，环境路径等
- `conda env list`：列出现有的所有现有环境以及他们的路径

**创建环境命令**
- `conda create -n $your_env_name python= $version_you_want`：创建一个虚拟环境并且指定其名字和python版本

**删除环境命令**
- `conda remove -n $your_env_name --all`

**启用虚拟环境**
- `conda activate $your_env_name`

**在当前虚拟环境下安装python包** 
- `pip install $package_name`