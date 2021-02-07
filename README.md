# Source code for Nature Coommunication manuscript "..."

## Introduction
This repository is used for publishing source code and the related usage in the Nature Coommunication manuscript "...".

## Requirement
* Ubuntu: 18.04
* Anaconda: 4.8.3
* pytorch: 1.6.0

## Usage
1. Load cell image data array.
2. Load model from `trained_model` directory.
3. Perform prediction.

### Labels for the trained model
Label | Description
0 | Astrocyte
1 | Oligodendrocyte
2 | Neuron

Detailed demo code is shown in `demo.py`, which utilizes the data in `example_data` directory.

## Citation
...(**Manuscript submitted**)