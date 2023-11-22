#!/bin/bash

# Create env
conda create -n ppiformer_demo python==3.9.17 -y
conda activate ppiformer_demo

# Install torch
pip install torch==1.13.1
pip install cmake
pip install git+https://github.com/pyg-team/pyg-lib.git
pip install torch_geometric==2.3.1
pip install torch-scatter torch-sparse
pip install pytorch_lightning==1.9.0

# Install dependencies
pip install git+https://github.com/a-r-j/graphein.git@master
pip install equiformer-pytorch==0.3.9
pip install wandb
pip install hydra-core
pip install -U hydra-submitit-launcher
pip install -e ../PPIRef
pip install -e ../mutils

# Install current project
pip install -e .
