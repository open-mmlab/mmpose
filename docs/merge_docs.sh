#!/usr/bin/env bash

sed -i '$a\\n' ../configs/bottom_up/*/*.md
sed -i '$a\\n' ../configs/top_down/*/*.md
sed -i '$a\\n' ../configs/wholebody/*/*.md
sed -i '$a\\n' ../configs/hand/*/*.md
sed -i '$a\\n' ../configs/face/*/*.md
sed -i '$a\\n' ../configs/mesh/*/*.md
sed -i '$a\\n' ../configs/fashion/*/*.md
sed -i '$a\\n' ../configs/animal/*/*.md
sed -i '$a\\n' ../demo/docs/*_demo.md

cat ../configs/bottom_up/*/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Bottom Up Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >bottom_up_models.md
cat ../configs/top_down/*/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Top Down Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >top_down_models.md
cat ../configs/wholebody/*/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Whole-Body Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >wholebody_models.md
cat ../configs/hand/*/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Hand Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >hand_models.md
cat ../configs/face/*/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Face Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >face_models.md
cat ../configs/mesh/*/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Mesh Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >mesh_models.md
cat ../configs/fashion/*/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Fashion Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >fashion_models.md
cat ../configs/animal/*/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Animal Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >animal_models.md
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
