#!/bin/bash

docker run -v $(pwd):/work/jec-gnn -w /work/jec-gnn --user $(id -u):$(id -g) --gpus all -it jec-gnn
