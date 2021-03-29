FROM  pytorch/pytorch

COPY ./ /workspace

WORKDIR /workspace

RUN python setup.py install
