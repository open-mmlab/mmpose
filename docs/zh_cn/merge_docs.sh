#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

sed -i '$a\\n' ../../demo/docs/*_demo.md
cat ../../demo/docs/*_demo.md | sed "s/^## 2D\(.*\)Demo/##\1Estimation/" | sed "s/md###t/html#t/g" | sed '1i\# Demos\n' | sed 's=](/docs/en/=](/=g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' >demos.md

 # remove /docs/ for link used in doc site
sed -i 's=](/docs/zh_cn/=](=g' overview.md
sed -i 's=](/docs/zh_cn/=](=g' installation.md
sed -i 's=](/docs/zh_cn/=](=g' quick_run.md
sed -i 's=](/docs/zh_cn/=](=g' migration.md
sed -i 's=](/docs/zh_cn/=](=g' ./model_zoo/*.md
sed -i 's=](/docs/zh_cn/=](=g' ./model_zoo_papers/*.md
sed -i 's=](/docs/zh_cn/=](=g' ./user_guides/*.md
sed -i 's=](/docs/zh_cn/=](=g' ./advanced_guides/*.md
sed -i 's=](/docs/zh_cn/=](=g' ./dataset_zoo/*.md
sed -i 's=](/docs/zh_cn/=](=g' ./notes/*.md
sed -i 's=](/docs/zh_cn/=](=g' ./projects/*.md


sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' overview.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' installation.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' quick_run.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' migration.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' ./advanced_guides/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' ./model_zoo/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' ./model_zoo_papers/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' ./user_guides/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' ./dataset_zoo/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' ./notes/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/main/=g' ./projects/*.md
