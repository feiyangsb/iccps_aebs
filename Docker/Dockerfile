FROM feiyangsb/carla_cuda:latest
LABEL maintainer "Feiyang Cai <feiyang.cai@vanderbilt.edu>"

USER root
RUN apt-get update 
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/gcc-7.1
RUN apt-get update 
RUN apt-get install -y --no-install-recommends gcc-7 g++-7
RUN apt-get install -y --no-install-recommends python3
RUN apt-get install -y --no-install-recommends python3-dev
RUN apt-get install -y --no-install-recommends python3-pip
RUN apt-get install -y --no-install-recommends python3-setuptools
RUN apt-get install -y --no-install-recommends libglib2.0-0
RUN apt-get install -y --no-install-recommends libsm6 libxext6 libxrender-dev
RUN apt-get install -y --no-install-recommends libpng16-16
RUN apt-get install -y --no-install-recommends libjpeg-turbo8
RUN apt-get install -y --no-install-recommends libtiff5

RUN pip3 install --upgrade pip
#RUN pip3 install numpy==1.16.4 
RUN pip3 install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install tensorflow-gpu==1.14.0
RUN pip3 install keras
RUN pip3 install psutil
RUN pip3 install pygame 
RUN pip3 install opencv-python

USER carla
RUN echo 'export CARLA_SERVER=/home/carla/CarlaUE4.sh' >> ~/.bashrc
RUN echo 'export PYTHONPATH=/home/carla/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg:$PYTHONPATH' >> ~/.bashrc
RUN echo 'export PYTHONDONTWRITEBYTECODE=1' >> ~/.bashrc
WORKDIR /home/carla/aebs
