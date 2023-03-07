# 如何给 MMPOSE 贡献代码

欢迎加入 MMPOSE 社区，我们致力于打造最前沿的计算机视觉基础库，我们欢迎任何形式的贡献，包括但不限于：

- **修复错误**
  1. 如果提交的代码改动较大，我们鼓励你先开一个 issue 并正确描述现象、原因和复现方式，讨论后确认修复方案。
  2. 修复错误并补充相应的单元测试，提交 PR 。
- **新增功能或组件**
  1. 如果新功能或模块涉及较大的代码改动，我们建议先提交 issue，与我们确认功能的必要性。
  2. 实现新增功能并添加单元测试，提交 PR 。
- **文档补充或翻译**
  - 如果发现文档有错误或不完善的地方，欢迎直接提交 PR 。

```{note}
- 如果你希望向 MMPose 1.0 贡献代码，请从 dev-1.x 上创建新分支，并提交 PR 到 dev-1.x 分支上。
- 如果你是论文作者，并希望将你的方法加入到 MMPose 中，欢迎联系我们，我们将非常感谢你的贡献。
- 如果你希望尽快将你的项目分享到 MMPose 开源社区，欢迎将 PR 提到 Projects 目录下，该目录下的项目将简化 Review 流程并尽快合入。
- 如果你希望加入 MMPose 的维护者，欢迎联系我们，我们将邀请你加入 MMPose 的维护者群。
```

## 准备工作

PR 操作所使用的命令都是用 Git 去实现的，该章节将介绍如何进行 Git 配置与 GitHub 绑定。

### 1. Git 配置

首先，你需要在本地安装 Git，然后配置你的 Git 用户名和邮箱：

```Shell
# 在命令提示符（cmd）或终端（terminal）中输入以下命令，查看 Git 版本
git --version
```

然后，你需要检查自己的 Git Config 是否正确配置：

```Shell
# 在命令提示符（cmd）或终端（terminal）中输入以下命令，查看 Git 配置
git config --global --list
```

如果 `user.name` 和 `user.email` 为空，你需要配置你的 Git 用户名和邮箱：

```Shell
git config --global user.name "这里填入你的用户名"
git config --global user.email "这里填入你的邮箱"
```

最后，在 `git bash` 或者终端中输入以下命令生成 SSH key。生成成功后，会在用户目录下生成 `.ssh` 文件夹，里面包含 `id_rsa` 和 `id_rsa.pub` 两个文件，分别是私钥和公钥。

```Shell
# usermail 为你的 GitHub 邮箱
ssh-keygen -t rsa -C "usermail"
```

## 2. GitHub 绑定

首先，用记事本打开 `id_rsa.pub` 公钥文件，复制里面的全部内容。

然后，登录你的 GitHub 账号，打开 `Settings`，在左侧菜单栏中点击 `SSH and GPG keys`，点击右上角的 `New SSH key`。

![](https://user-images.githubusercontent.com/90811472/221778382-a075167d-b028-4f68-a1c7-49a8f6f3d97b.png)

在 `Title` 中填入一个你自己能识别的名字，比如 `git`，在 `Key` 中粘贴你刚刚复制的公钥，点击 `Add SSH key`，添加成功后，你就可以使用 SSH 连接 GitHub 了。

![](https://user-images.githubusercontent.com/90811472/221549754-53670c19-5efe-48b2-9ac5-bafb43891903.png)

最后，在 `git bash` 或者终端中输入以下命令，测试是否配置成功：

```shell
ssh -T git@github.com
```

如果出现以下提示，说明你已经成功配置了 Git 和 GitHub：

```shell
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

## PR 流程

如果你对 PR 流程不熟悉，接下来将会从零开始，一步一步地教你如何提交 PR。如果你想深入了解 PR 开发模式，可以参考 [GitHub 官方文档](https://docs.github.com/cn/github/collaborating-with-issues-and-pull-requests/about-pull-requests)。

### 1. Fork 项目

当你第一次提交 PR 时，需要先 Fork 项目到自己的 GitHub 账号下。点击项目右上角的 Fork 按钮，将项目 Fork 到自己的 GitHub 账号下。

![](https://user-images.githubusercontent.com/13503330/223318144-a49c6cef-b1fb-45b8-aa2b-0833d0e3fd5c.png)

接着，你需要将你的 Fork 仓库 Clone 到本地，然后添加官方仓库作为远程仓库：

```Shell

# Clone 你的 Fork 仓库到本地
git clone git@github.com:{username}/mmpose.git

# 添加官方仓库作为远程仓库
cd mmpose
git remote add upstream git@github.com:open-mmlab/mmpose.git
```

在终端中输入以下命令，查看远程仓库是否成功添加：

```Shell
git remote -v
```

如果出现以下信息，说明你已经成功添加了远程仓库：

```Shell
origin	git@github.com:{username}/mmpose.git (fetch)
origin	git@github.com:{username}/mmpose.git (push)
upstream	git@github.com:open-mmlab/mmpose (fetch)
upstream	git@github.com:open-mmlab/mmpose (push)
```

```{note}
这里对 origin 和 upstream 进行一个简单的介绍，当我们使用 git clone 来克隆代码时，会默认创建一个 origin 的 remote，它指向我们克隆的代码库地址，而 upstream 则是我们自己添加的，用来指向原始代码库地址。当然如果你不喜欢他叫 upstream，也可以自己修改，比如叫 open-mmlab。我们通常向 origin 提交代码（即 fork 下来的远程仓库），然后向 upstream 提交一个 pull request。如果提交的代码和最新的代码发生冲突，再从 upstream 拉取最新的代码，和本地分支解决冲突，再提交到 origin。
```

### 2. 配置 pre-commit

在本地开发环境中，我们使用 pre-commit 来检查代码风格，以确保代码风格的统一。在提交代码前，你需要先安装 pre-commit：

```Shell
pip install -U pre-commit

# 在 mmpose 根目录下安装 pre-commit
pre-commit install
```

检查 pre-commit 是否配置成功，并安装 `.pre-commit-config.yaml` 中的钩子：

```Shell
pre-commit run --all-files
```

![](https://user-images.githubusercontent.com/57566630/202368856-0465a90d-8fce-4345-918e-67b8b9c82614.png)

```{note}
如果你是中国大陆用户，由于网络原因，可能会出现 pre-commit 安装失败的情况。

这时你可以使用清华源来安装 pre-commit：
pip install -U pre-commit -i https://pypi.tuna.tsinghua.edu.cn/simple

或者使用国内镜像来安装 pre-commit：
pip install -U pre-commit -i https://pypi.mirrors.ustc.edu.cn/simple
```

如果安装过程被中断，可以重复执行上述命令，直到安装成功。

如果你提交的代码中有不符合规范的地方，pre-commit 会发出警告，并自动修复部分错误。

![](https://user-images.githubusercontent.com/57566630/202369176-67642454-0025-4023-a095-263529107aa3.png)

### 3. 创建开发分支

安装完 pre-commit 之后，我们需要基于 dev 分支创建一个新的开发分支，建议以 `username/pr_name` 的形式命名，例如：

```Shell
git checkout -b username/refactor_contributing_doc
```

在后续的开发中，如果本地仓库的 dev 分支落后于官方仓库的 dev 分支，需要先拉取 upstream 的 dev 分支，然后 rebase 到本地的开发分支上：

```Shell
# 在本地切换回 dev-1.x 分支
git checkout dev-1.x

# 拉取官方仓库的 dev-1.x 分支
git pull upstream dev-1.x

# 切换回开发分支
git checkout username/refactor_contributing_doc

# rebase dev 分支
git rebase dev-1.x
```

在 rebase 时，如果出现冲突，需要手动解决冲突，然后执行 `git add` 命令，再执行 `git rebase --continue` 命令，直到 rebase 完成。

### 4. 提交代码并在本地通过单元测试

在本地开发完成后，我们需要在本地通过单元测试，然后提交代码。

```shell
# 运行单元测试
pytest tests/

# 提交代码
git add .
git commit -m "commit message"
```

### 5. 推送代码到远程仓库

在本地开发完成后，我们需要将代码推送到远程仓库。

```Shell
git push origin username/refactor_contributing_doc
```

### 6. 提交 Pull Request (PR)

#### (1) 在 GitHub 上创建 PR

![](https://user-images.githubusercontent.com/13503330/223321382-e6068e18-1d91-4458-8328-b1c7c907b3b2.png)

#### (2) 在 PR 中根据指引修改描述，添加必要的信息

![](https://user-images.githubusercontent.com/13503330/223322447-94ad4b8c-21bf-4ca7-b3d6-0568cace6eee.png)

```{note}
- 在 PR branch 左侧选择 `dev` 分支，否则 PR 会被拒绝。
- 如果你是第一次向 OpenMMLab 提交 PR，需要签署 CLA。
```

![](https://user-images.githubusercontent.com/57566630/167307569-a794b967-6e28-4eac-a942-00deb657815f.png)

## 指引

### 单元测试

我们使用 [pytest](https://docs.pytest.org/en/latest/) 来进行单元测试。我们的单元测试文件存储在 `tests/` 文件夹下。

在提交修复代码或新增特性的 PR 时，我们应该尽可能地让单元测试覆盖所有提交的代码。如果你的 PR 仅仅是修改了文档，那么你可以跳过单元测试。

单元测试覆盖率计算方法如下：

```Shell
python -m coverage run -m pytest tests/
python -m coverage html
# check file `htmlcov/index.html` to see coverage
```

### 文档渲染

在提交修复代码或新增特性的 PR 时，可能会需要修改或新增模块的 docstring。我们使用 [sphinx](https://www.sphinx-doc.org/en/master/) 来渲染文档。通过如下方式可以本地生成渲染后的文档：

```Shell
pip install -r docs/requirements.txt

# cd docs/en or docs/zh_cn
cd docs/zh_cn

make html
# check file in ./docs/zh_cn/_build/html/index.html
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

```{note}
在你创建PR之前，请确保你的代码格式符合规范，且经过了 yapf 格式化。
```

### C++与CUDA

遵循[Google C++风格指南](https://google.github.io/styleguide/cppguide.html)
