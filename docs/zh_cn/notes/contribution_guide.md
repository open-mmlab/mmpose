# 参与贡献代码

我们欢迎任何形式的贡献，包括但不限于：

- 修复错别字、bug
- 增加文档内容或翻译
- 添加新功能和组件

## 流程

1. Fork MMPose官方代码仓库，并Pull最新的代码
2. 创建一个新的分支（请不要直接在 master 分支上进行开发）
3. 提交你的改动
4. 创建一个PR

```{note}
- 如果你希望向 MMPose 1.0 贡献代码，请从 dev-1.x 上创建新分支，并提交 PR 到 dev-1.x 分支上
- 如果你打算添加的新功能涉及的改动较大，我们鼓励你先开一个 issue 与我们进行讨论。
- 如果你是论文作者，并希望将你的方法加入到 MMPose 中，欢迎联系我们，我们将非常感谢你的贡献。
```

## 代码风格

### Python

我们采用[PEP8](https://www.python.org/dev/peps/pep-0008/)作为代码风格。

使用下面的工具来对代码进行整理和格式化：

- [flake8](http://flake8.pycqa.org/en/latest/)：代码提示
- [isort](https://github.com/timothycrosley/isort)：import 排序
- [yapf](https://github.com/google/yapf)：格式化工具
- [codespell](https://github.com/codespell-project/codespell)： 单词拼写检查
- [mdformat](https://github.com/executablebooks/mdformat): markdown 文件格式化工具
- [docformatter](https://github.com/myint/docformatter): docstring 格式化工具

`yapf`和`isort`的样式配置可以在[setup.cfg](/setup.cfg)中找到。

我们使用[pre-commit hook](https://pre-commit.com/)来：

- 检查和格式化 `flake8`、`yapf`、`isort`、`trailing whitespaces`
- 修复 `end-of-files`
- 在每次提交时自动排序 `requirments.txt`

`pre-commit`的配置存储在[.pre-commit-config](/.pre-commit-config.yaml)中。

在clone代码仓库后，你需要安装并初始化 `pre-commit`：

```Shell
pip install -U pre-commit
```

并在 MMPose 仓库目录下运行：

```shell
pre-commit install
```

在顺利安装后，你每次提交代码时都会自动执行代码格式检查与自动格式化。

```{note}
在你创建PR之前，请确保你的代码格式符合规范，且经过了yapf格式化
```

### C++与CUDA

遵循[Google C++风格指南](https://google.github.io/styleguide/cppguide.html)
