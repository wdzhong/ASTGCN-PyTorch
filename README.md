# ASTGCN-PyTorch

Rewrite the [MXNet](https://mxnet.incubator.apache.org) version [ASTGCN](https://github.com/Davidham3/ASTGCN) using [PyTorch](https://pytorch.org/). Please refer to the MXNet version for more details.

## ATTENTION: This repo is still under active development.
The program is runnable, but the training result is **_NOT_** correct. Issues and/or pull requests are welcome.

## Requirements
- PyTorch >= 1.1.0
- SciPy


## Usage
Train model on PEMS04:
```
    python train.py --config configurations/PEMS04.conf --force True
```
