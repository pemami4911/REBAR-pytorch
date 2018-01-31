# **WORK IN PROGRESS**

## REBAR-pytorch
Implementation and replication of experiments for the [REBAR paper](https://papers.nips.cc/paper/6856-rebar-low-variance-unbiased-gradient-estimates-for-discrete-latent-variable-models.pdf) from NIPS 2017 in PyTorch

Tensorflow implementation by the authors [here](https://github.com/tensorflow/models/tree/master/research/rebar)

For now, I have implemented REBAR for the toy problem in Section 5.1. See `rebar_toy.ipynb`.

## Implementation plan

* [X] Data preparation - add data_download.py, datasets.py, and config.py from authors implementation to this repo
* [ ] rebar_train - create as Jupyter notebook, with train loop and plotting of results
* [ ] rebar.py - models
