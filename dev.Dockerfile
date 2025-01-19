FROM python:3.12

WORKDIR /src

RUN pip3 install torch torchvision torchaudio

COPY ./requirements.txt ./

RUN pip install -U -r ./requirements.txt