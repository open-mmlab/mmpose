#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

sed -i '$a\\n' ../../demo/docs/*_demo.md
cat ../../demo/docs/*_demo.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Demo' | sed 's=](/docs/en/=](/=g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >demo.md

 # remove /docs/ for link used in doc site
sed -i 's=](/docs/en/=](=g' ./tutorials/*.md
sed -i 's=](/docs/en/=](=g' ./tasks/*.md
sed -i 's=](/docs/en/=](=g' ./papers/*.md
sed -i 's=](/docs/en/=](=g' ./topics/*.md
sed -i 's=](/docs/en/=](=g' data_preparation.md
sed -i 's=](/docs/en/=](=g' getting_started.md
sed -i 's=](/docs/en/=](=g' install.md
sed -i 's=](/docs/en/=](=g' benchmark.md
sed -i 's=](/docs/en/=](=g' changelog.md
sed -i 's=](/docs/en/=](=g' faq.md

sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tutorials/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tasks/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./papers/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./topics/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' data_preparation.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' getting_started.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' install.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' benchmark.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' changelog.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' faq.md
