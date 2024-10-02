在 `git clone` 时使用代理，你可以通过以下几种方法来配置 Git 代理。

### 1. 临时设置代理
如果你只想在当前操作中使用代理，可以在命令中通过 `http.proxy` 参数指定代理。

```bash
git -c http.proxy=http://proxy-server:port clone https://github.com/username/repository.git
```

其中，`proxy-server` 是你的代理服务器地址，`port` 是代理服务器的端口号。

### 2. 全局设置代理
如果你想在 Git 的所有操作中使用代理，可以通过设置 Git 的配置文件来实现。你可以选择全局代理（适用于所有仓库）或局部代理（只适用于当前仓库）。

#### 设置全局代理
使用以下命令配置全局代理：

```bash
git config --global http.proxy http://proxy-server:port
git config --global https.proxy http://proxy-server:port
```

#### 设置局部代理
如果你只想在某个仓库中使用代理，可以在该仓库中运行以下命令：

```bash
git config http.proxy http://proxy-server:port
git config https.proxy http://proxy-server:port
```

### 3. 删除代理设置
如果你不再需要代理，可以通过以下命令删除代理设置：

#### 删除全局代理
```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

#### 删除局部代理
```bash
git config --unset http.proxy
git config --unset https.proxy
```

通过这些方法，你就可以在使用 Git 时配置走代理。如果代理需要用户名和密码，可以通过以下格式配置代理：

```bash
http://username:password@proxy-server:port
```