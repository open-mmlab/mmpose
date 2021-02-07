#!/usr/bin/env bash

sed -i '$a\\n' ../configs/bottom_up/*/*.md
sed -i '$a\\n' ../configs/top_down/*/*.md
sed -i '$a\\n' ../demo/*_demo.md
sed -i '$a\\n' ../configs/hand/*/*.md
sed -i '$a\\n' ../configs/face/*/*.md
sed -i '$a\\n' ../configs/mesh/*/*.md
sed -i '$a\\n' ../configs/fashion/*/*.md

cat ../configs/bottom_up/*/*.md >bottom_up_models.md
cat ../configs/top_down/*/*.md >top_down_models.md
cat ../demo/*_demo.md >demo.md
cat ../configs/hand/*/*.md >hand_models.md
cat ../configs/face/*/*.md >face_models.md
cat ../configs/mesh/*/*.md >mesh_models.md
cat ../configs/fashion/*/*.md >fashion_models.md

sed -i "s/#/#&/" bottom_up_models.md
sed -i "s/#/#&/" top_down_models.md
sed -i "s/#/#&/" demo.md
sed -i "s/#/#&/" hand_models.md
sed -i "s/#/#&/" face_models.md
sed -i "s/#/#&/" mesh_models.md
sed -i "s/#/#&/" fashion_models.md
sed -i "s/md###t/html#t/g" bottom_up_models.md
sed -i "s/md###t/html#t/g" top_down_models.md
sed -i "s/md###t/html#t/g" demo.md
sed -i "s/md###t/html#t/g" hand_models.md
sed -i "s/md###t/html#t/g" face_models.md
sed -i "s/md###t/html#t/g" mesh_models.md
sed -i "s/md###t/html#t/g" fashion_models.md

sed -i '1i\# Bottom Up Models' bottom_up_models.md
sed -i '1i\# Top Down Models' top_down_models.md
sed -i '1i\# Demo' demo.md
sed -i '1i\# Hand Models' hand_models.md
sed -i '1i\# Face Models' face_models.md
sed -i '1i\# Mesh Models' mesh_models.md
sed -i '1i\# Fashion Models' fashion_models.md

sed -i 's/](\/docs\//](/g' bottom_up_models.md # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' top_down_models.md
sed -i 's/](\/docs\//](/g' hand_models.md
sed -i 's/](\/docs\//](/g' face_models.md
sed -i 's/](\/docs\//](/g' mesh_models.md
sed -i 's/](\/docs\//](/g' fashion_models.md
sed -i 's/](\/docs\//](/g' ./tutorials/*.md
sed -i 's/](\/docs\//](/g' data_preparation.md
sed -i 's/](\/docs\//](/g' ./tasks/*.md

sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' bottom_up_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' top_down_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' hand_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' face_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' mesh_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' fashion_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' getting_started.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' install.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' benchmark.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' changelog.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' demo.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' faq.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tutorials/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' data_preparation.md
sed -i 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' ./tasks/*.md
