# Zero-One Laws of Graph Neural Networks

This repository contains the code for the experiments in the paper [Zero-One Laws of Graph Neural Networks](https://arxiv.org/abs/2301.13060) by Sam Adam-Day, Theodor Mihai Iliant and İsmail İlkan Ceylan. See the paper for more details on the experimental setup.


## Repository structure

The experiments are located in a Jupyter notebooks, as follows. 

- [`GCN.ipynb`](/GCN.ipynb): All experiments with graph convolutional networks (GCNs), including additional experiments found in the appendix:
    * Starting with normally distributed initial node features.
    * Using other non-linearities.
    * Using different graph distributions.
- [`MeanGNN.ipynb`](/MeanGNN.ipynb): Main experiments with message-passing neural networks (MPNNs) with mean aggregation.
- [`SumGNN.ipynb`](/SumGNN.ipynb): Main experiments with MPNNs with sum aggregation.

The [`additional-experiments`](/additional-experiments/) folder contains some further experiments not reported in the paper.

Graphs of the experiment results are found in [`visuals`](/visuals/), and experiment artifacts are located in [`saved-models`](/saved-models/) and [`pickle-images`](/pickle-images/).


## Experimental details

Since we randomly initialize the graph neural networks involved in the experiments (to validate our theorems which allow for an arbitrary such network), there is no training/evaluation. Test data consists of the graphs that are fed to these graph neural networks, which are generated according to the Erdős–Rényi model in the experiments that validate our proved theorems.


## Running the experiments

Notebooks are intended to be run in [Google Colab](https://colab.research.google.com/), but can be adapted straightforwardly to be run elsewhere. The requirements are listed in [`requirements.txt`](/requirements.txt).
