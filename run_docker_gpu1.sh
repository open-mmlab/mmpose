docker run --network host -w /data/new_mmpose/mmpose \
  -v /data:/data \
  --gpus '"device=1"' --shm-size=8g -it mmpose:1.3.2
