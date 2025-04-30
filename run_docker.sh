docker run --network host -w /data/mmpose \
  -v /data:/data \
  --gpus all --shm-size=8g -it mmpose \
  /bin/bash -c "python setup.py develop && /bin/bash"
