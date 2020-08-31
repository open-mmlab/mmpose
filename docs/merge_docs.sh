#!/usr/bin/env bash

sed -i '$a\\n' ../configs/bottom_up/*/*.md
sed -i '$a\\n' ../configs/top_down/*/*.md
sed -i '$a\\n' pretrained.md

cat  ../configs/bottom_up/*/*.md > bottom_up_models.md
cat  ../configs/top_down/*/*.md > top_down_models.md

sed -i "s/#/#&/" bottom_up_models.md
sed -i "s/#/#&/" top_down_models.md
sed -i "s/md###t/html#t/g" bottom_up_models.md
sed -i "s/md###t/html#t/g" top_down_models.md

sed -i '1i\# Bottom Up Models' bottom_up_models.md
sed -i '1i\# Top Down Models' top_down_models.md

sed -i 's/](\/docs\//](/g' bottom_up_models.md # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' top_down_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' bottom_up_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' top_down_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' getting_started.md
