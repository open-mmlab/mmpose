#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

sed -i '$a\\n' ../demo/docs/*_demo.md
cat ../demo/docs/*_demo.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# 示例' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >demo.md

 # remove /docs_zh-CN/ for link used in doc site
sed -i 's/](\/docs_zh-CN\//](/g' ./tutorials/*.md
sed -i 's/](\/docs_zh-CN\//](/g' ./tasks/*.md
sed -i 's/](\/docs_zh-CN\//](/g' ./papers/*.md
sed -i 's/](\/docs_zh-CN\//](/g' ./topics/*.md
sed -i 's/](\/docs_zh-CN\//](/g' data_preparation.md
sed -i 's/](\/docs_zh-CN\//](/g' getting_started.md
sed -i 's/](\/docs_zh-CN\//](/g' install.md
sed -i 's/](\/docs_zh-CN\//](/g' benchmark.md
# sed -i 's/](\/docs_zh-CN\//](/g' changelog.md
sed -i 's/](\/docs_zh-CN\//](/g' faq.md

sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tutorials/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tasks/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./papers/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./topics/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' data_preparation.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' getting_started.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' install.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' benchmark.md
# sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' changelog.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' faq.md
