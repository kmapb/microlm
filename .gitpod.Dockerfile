FROM gitpod/workspace-full

RUN sudo apt-get update
RUN sudo apt-get install -y libvulkan1 libgl1 libglib2.0-0 wget
# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
RUN sudo apt-get install -y --fix-missing nvidia-cuda-toolkit

## This really does seem to need to be an && pipeline(?)
RUN \
    wget https://github.com/Juice-Labs/Juice-Labs/releases/latest/download/JuiceClient-linux.tar.gz && \
    sudo mkdir /JuiceClient && \
    sudo mv JuiceClient-linux.tar.gz /JuiceClient

WORKDIR /JuiceClient
RUN sudo tar xvzf JuiceClient-linux.tar.gz
RUN sudo chown -R gitpod /JuiceClient
ENV JUICE_CFG_OVERRIDE '{"host": "100.108.145.17"}'

## Install tailscale
RUN curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/focal.gpg | sudo apt-key add - \
     && curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/focal.list | sudo tee /etc/apt/sources.list.d/tailscale.list \
     && sudo apt-get update -q \
     && sudo apt-get install -yq tailscale jq \
     && sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-nft
