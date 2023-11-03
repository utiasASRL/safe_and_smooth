#FROM python:3.10-bookworm
FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y sudo git-core bash-completion \
		&& apt-get install -y python3-pip \
		&& pip3 install --upgrade pip
RUN rm -rf /var/lib/apt/lists/*

RUN mkdir ~/.ssh
ARG ssh_prv_key
ARG ssh_pub_key
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    pip install git+https://github.com/utiasASRL/poly_matrix && \
    pip install git+https://github.com/utiasASRL/safe_and_smooth && \
    rm /root/.ssh/id_rsa*

RUN mkdir /home/user
WORKDIR /home/user/safe_and_smooth
CMD ["bash"]
