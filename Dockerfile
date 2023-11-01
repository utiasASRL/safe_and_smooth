FROM ubuntu:20.04

# Install system requirements
RUN apt update && apt install -y python3 python3-pip libsuitesparse-dev git && rm -rf /var/lib/apt/lists/* 

# Install dependencies:
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN git clone --recursive https://github.com/utiasASRL/safe_and_smooth && cd safe_and_smooth && pip install -e .

# Run the application:
WORKDIR "/safe"
