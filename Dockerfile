FROM python:3.12-slim AS biosspheres-notebook
LABEL description="Dockerize biosspheres for reproducibility purposes with jupyterlab"

WORKDIR /root
COPY . /root/biosspheres
RUN pip install --no-cache-dir "/root/biosspheres[jupyter]"

EXPOSE 8888/tcp
ENV SHELL=/bin/bash
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
