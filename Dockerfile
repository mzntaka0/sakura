FROM  pytorch/pytorch
RUN pip uninstall sakura

WORKDIR /workspace

RUN apt update -y && apt install git -y
RUN pip install git+https://github.com/zakuro-ai/sakura.git
