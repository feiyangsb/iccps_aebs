#!/bin/sh
sudo chown -R 1001:1000 ./
if [ -z $(docker ps -q -f name=carla_env) ]; then
    echo "starting carla docker...";
    docker run -it --rm --runtime=nvidia \
    --name carla_env \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -u carla \
    -v $(pwd)/:/home/carla/aebs \
    --net="host" \
    feiyangsb/carla_aebs:latest \
    /bin/bash;
else
    echo "attaching a new terminal...";
    docker exec -it carla_env bash;
fi