#!/usr/bin/env bash

sed -i '$a\\n' ../configs/animal/*/*/_result_collection/*.md
sed -i '$a\\n' ../configs/body/*/*/_result_collection/*.md
sed -i '$a\\n' ../configs/face/*/*/_result_collection/*.md
sed -i '$a\\n' ../configs/fashion/*/*/_result_collection/*.md
sed -i '$a\\n' ../configs/hand/*/*/_result_collection/*.md
sed -i '$a\\n' ../configs/wholebody/*/*/_result_collection/*.md

cat ../configs/animal/*/*/_result_collection/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Animal Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >animal_models.md
cat ../configs/body/*/*/_result_collection/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Body Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >body_models.md
cat ../configs/face/*/*/_result_collection/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Face Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >face_models.md
cat ../configs/fashion/*/*/_result_collection/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Fashion Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >fashion_models.md
cat ../configs/hand/*/*/_result_collection/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# Hand Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >hand_models.md
cat ../configs/wholebody/*/*/_result_collection/*.md | sed "s/#/#&/" | sed "s/md###t/html#t/g" | sed '1i\# WholeBody Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmpose/tree/master/=g' >wholebody_models.md

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
