FROM pytorch/torchserve:latest-gpu

COPY . /home/model-server
WORKDIR /home/model-server

RUN 

CMD ["torchserve", "--start", "--model-store", "/home/model-server/model-store", "--models", "model.mar"]