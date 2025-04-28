docker run --network host -w /mmpose \
  -dit \
  --name mmpose \
  -v /code/mmpose:/mmpose \
  -v /data:/data \
  --gpus '"device=0"' --shm-size=8g -it \
  mmpose:latest

