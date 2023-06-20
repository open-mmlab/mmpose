docker run -w /data/mmpose \
  -v /data:/data \
  --gpus all --shm-size=8g -it mmpose