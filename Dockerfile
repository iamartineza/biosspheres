FROM ubuntu:latest as biosspheres-notebook
LABEL description="Dockerize biosspheres for reproducibility purposes with jupyterlab"

COPY requirements.txt /root
WORKDIR /root

RUN apt-get update && apt-get install -y sudo && rm -rf /var/lib/apt/lists/*
RUN sudo apt update -y 
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt

EXPOSE 8888/tcp
ENV SHELL /bin/bash
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
