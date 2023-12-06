IMG=zebincai/nerf:latest
REPO_NAME=nrr
NAME=zebin_${REPO_NAME}

docker run -it -d --name $NAME \
  --gpus all \
  --privileged \
  --hostname in_docker \
  --add-host in_docker:127.0.0.1 \
  --add-host $(hostname):127.0.0.1 \
  --shm-size 2G \
  -e DISPLAY \
  -p 6005:22 \
  -v /etc/localtime:/etc/localtime:ro \
  -v /media:/media \
  -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/zebin/work/nerf/Instant-angelo:/${REPO_NAME} \
  -w /${REPO_NAME} \
  $IMG \
  /bin/bash
