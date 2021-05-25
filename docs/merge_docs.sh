#!/usr/bin/env bash

sed -i '$a\\n' ../configs/README/datasets/*.md
sed -i '$a\\n' ../configs/README/techniques/*.md

cat ../configs/README/datasets/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Datasets' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >datasets_models.md
cat ../configs/README/techniques/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Techniques' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >techniques_models.md

sed -i '$a\\n' ../demo/docs/*_demo.md
cat ../demo/docs/*_demo.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Demo' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >demo.md

 # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' ./tutorials/*.md
sed -i 's/](\/docs\//](/g' ./tasks/*.md
sed -i 's/](\/docs\//](/g' data_preparation.md
sed -i 's/](\/docs\//](/g' getting_started.md
sed -i 's/](\/docs\//](/g' install.md
sed -i 's/](\/docs\//](/g' benchmark.md
sed -i 's/](\/docs\//](/g' changelog.md
sed -i 's/](\/docs\//](/g' faq.md

sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tutorials/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tasks/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' data_preparation.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' getting_started.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' install.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' benchmark.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' changelog.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' faq.md
