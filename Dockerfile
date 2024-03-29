#FROM python:3.10-bookworm
FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y sudo git-core bash-completion  && \
		apt-get install -y python3-pip && \
		pip3 install --upgrade pip && \
		pip3 install pytest
RUN rm -rf /var/lib/apt/lists/*

RUN mkdir ~/.ssh
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    pip3 install git+https://github.com/utiasASRL/poly_matrix && \
    pip3 install git+https://github.com/utiasASRL/safe_and_smooth

RUN mkdir /home/user
WORKDIR /home/user/safe_and_smooth
CMD ["bash"]
