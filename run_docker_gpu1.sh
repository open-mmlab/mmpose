docker run --network host -w /mmpose \
  -v /data/mmpose:/mmpose \
  --gpus '"device=1"' --shm-size=8g -it mmpose:1.3.2
