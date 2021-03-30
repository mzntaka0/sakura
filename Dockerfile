FROM  pytorch/pytorch

WORKDIR /workspace

RUN apt update -y && apt install git -y
RUN pip install git+https://github.com/zakuro-ai/sakura.git
