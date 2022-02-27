FROM tensorflow/tensorflow:2.4.1-gpu

WORKDIR /work

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install graphviz -y

USER dev

ENV PATH=/work/jec-gnn/bin:$PATH
ENV PYTHONPATH=/work/jec-gnn