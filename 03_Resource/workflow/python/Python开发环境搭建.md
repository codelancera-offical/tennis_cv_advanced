---
type: knowledge_card
tags:
  - python
  - blog
date_creation: 2024-07-10 16:52
---
```dataview
TABLE dateformat(this.file.mtime, "yyyy-MM-dd HH:mm:ss") AS "Last Modified Time"
WHERE file.path = this.file.path
```

# Python开发环境搭建

## Anaconda配置
1. **安装Anaconda**
    - [官网下载](https://www.anaconda.com/download?utm_source=anacondadoc&utm_medium=documentation&utm_campaign=download&utm_content=topnavalldocs)
    - 安装设置：
        - For all user
        - 设置安装路径
        - base环境默认为3.12
2. **添加环境变量**
    - `\anaconda3\Scripts`
    - `\anaconda3\Library\bin`
    - `\anaconda3
3. 在powershell中激活conda环境
    - 第一次运行： `conda init powershell`
4. anaconda换源（有魔法无所谓）
```conda
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/  
conda config --set show_channel_urls yes
```
5. 更改anaconda虚拟环境默认安装位置
```conda
conda config --add envs_dirs E:\30752\develop_softwares\anaconda3\envs
```
6. 把默认环境安装位置的用户权限拉满![[Pasted image 20240710164504.png]]
到这里anaconda的环境就基本配好了


## Vscode环境配置
1. **安装Vscode**
    - [Vscode安装链接](https://code.visualstudio.com/)
2. **在Vscode中安装python开发套件**
3. **选择python的解释器**
    - 在vscode右下角选择python的解释器为conda的python解释器

做完这些基本上python文件就可以运行以及调试了，直接右上角的播放按钮就可以选择运行或者调试。
