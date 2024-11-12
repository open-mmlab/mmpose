docker run --network host -w /data/mmpose \
  -v /data:/data \
  --gpus '"device=0"' --shm-size=8g -it mmpose \
  /bin/bash -c "python setup.py develop && /bin/bash"
/data/mmpose/work_dirs/old4