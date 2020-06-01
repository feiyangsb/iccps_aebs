#!/bin/sh
echo Building feiyangsb/carla_cuda:latest

docker build -t feiyangsb/carla_cuda . -f ./Docker/carla.Dockerfile

echo Building feiyangsb/carla_aebs:latest
docker build -t feiyangsb/carla_aebs . -f ./Docker/Dockerfile
