FROM gitpod/workspace-full

RUN sudo apt-get update
RUN sudo apt-get install -y libvulkan1 libgl1 libglib2.0-0 wget
RUN sudo apt-get install -y nvidia-cuda-toolkit

## This really does seem to need to be an && pipeline(?)
RUN \
    wget https://github.com/Juice-Labs/Juice-Labs/releases/latest/download/JuiceClient-linux.tar.gz && \
    mkdir /tmp/JuiceClient && \
    tar -xf JuiceClient-linux.tar.gz -C /tmp/JuiceClient

# WORKDIR /tmp/JuiceClient

# RUN tar xvzf JuiceClient-linux.tar.gz
ENV JUICE_CFG_OVERRIDE '{"host": "100.108.145.17"}'
