FROM jcadic/vanilla:torch3.6

COPY . /asyncdnn

WORKDIR /asyncdnn

RUN python setup.py install

CMD ["python", "-m", "asyncdnn.test"]
