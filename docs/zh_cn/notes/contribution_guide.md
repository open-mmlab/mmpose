# 参与贡献代码

我们欢迎任何形式的贡献，包括但不限于：

- 修复错别字、bug

- 添加新feature和组件

## 流程

1. Fork MMPose官方代码仓库，并Pull最新的代码

2. 创建一个新的分支，使用有具体意义的分支名（不要在master分支上做PR）

3. 提交你的改动

4. 创建一个PR

```{note}
- 如果你希望向 MMPose 1.0 贡献代码，请从dev-1.x上创建新分支，并提交PR到dev-1.x分支上
- 如果你打算添加的新feature涉及的改动较大，我们鼓励你先开一个issue与我们进行讨论。
- 如果你是论文作者，并希望将你的方法加入到MMPose中，欢迎联系我们，我们将非常感谢你的贡献。
```

## 代码风格

### Python

我们采用[PEP8](https://www.python.org/dev/peps/pep-0008/)作为代码风格。

使用下面的工具来对代码进行整理和格式化：

- [flake8](http://flake8.pycqa.org/en/latest/)：代码提示

- [yapf](https://github.com/google/yapf)：格式化工具

- [isort](https://github.com/timothycrosley/isort)：import排序

`yapf`和`isort`的样式配置可以在[setup.cfg](./setup.cfg)中找到。

我们使用[pre-commit hook](https://pre-commit.com/)来：

- 检查和格式化`flake8`、`yapf`、`isort`、`trailing whitespaces`

- 修复`end-of-files`

- 在每次提交时自动排序`requirments.txt`

`pre-commit`的配置存储在[.pre-commit-config](.../.pre-commit-config.yaml)中。

在clone代码仓库后，你需要安装并初始化`pre-commit`：

```Shell
pip install -U pre-commit
```

如果你在安装`markdown lint`时遇到问题，你可以通过以下方式安装`ruby for markdown lint`：

- 参考 [这个 repo](https://github.com/innerlee/setup)，按照指引并使用[zzruby.sh](https://github.com/innerlee/setup/blob/master/zzruby.sh)

或者通过以下步骤：

```Shell
# install rvm
curl -L https://get.rvm.io | bash -s -- --autolibs=read-fail
rvm autolibs disable
# install ruby
rvm install 2.7.1
```

在顺利安装后，你每次提交代码时都会强制执行代码格式检查与自动格式化。

```{note}
在你创建PR之前，请确保你的代码格式符合规范，且经过了yapf格式化
```

### C++与CUDA

遵循[Google C++风格指南](https://google.github.io/styleguide/cppguide.html)
