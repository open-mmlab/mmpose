# Contributing to MMPose

All kinds of contributions are welcome, including but not limited to the following.

- Fixes (typo, bugs)
- New features and components

## Workflow

1. Fork and pull the latest mmpose
1. Checkout a new branch with a meaningful name (do not use master branch for PRs)
1. Commit your changes
1. Create a PR

```{note}
- If you plan to add some new features that involve large changes, it is encouraged to open an issue for discussion first.
- If you are the author of some papers and would like to include your method to mmpose, please contact us. We will much appreciate your contribution.
```

## Code style

### Python

We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:

- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [yapf](https://github.com/google/yapf): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

Style configurations of yapf and isort can be found in [setup.cfg](../setup.cfg).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `isort`, `trailing whitespaces`,
fixes `end-of-files`, sorts `requirments.txt` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](../.pre-commit-config.yaml).

After you clone the repository, you will need to install initialize pre-commit hook.

```
pip install -U pre-commit
```

From the repository folder

```
pre-commit install
```

If you are facing an issue when installing markdown lint, you may install ruby for markdown lint by
referring to [this repo](https://github.com/innerlee/setup) by following the usage and taking [`zzruby.sh`](https://github.com/innerlee/setup/blob/master/zzruby.sh)

 or by the following steps

 ```shell
# install rvm
curl -L https://get.rvm.io | bash -s -- --autolibs=read-fail
rvm autolibs disable
 # install ruby
rvm install 2.7.1
```

After this on every commit check code linters and formatter will be enforced.

> Before you create a PR, make sure that your code lints and is formatted by yapf.

### C++ and CUDA

We follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
