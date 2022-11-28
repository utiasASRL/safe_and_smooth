FROM ubuntu:20.04

# Install system requirements
RUN apt update && apt install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run the application:
WORKDIR "/safe"
