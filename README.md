# Source code for manuscript "Deep learning-based predictive identification of neural stem cell differentiation"
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4606918.svg)](https://doi.org/10.5281/zenodo.4606918)

## Introduction
This repository is used for publishing source code and the related usage in the manuscript "Deep learning-based predictive identification of neural stem cell differentiation".

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
-----------|------------
0 | Astrocyte
1 | Oligodendrocyte
2 | Neuron

Detailed demo code is shown in `demo.py`, which utilizes the data in `example_data` directory.

## Citation
Yanjing Zhu, Rongrong Zhu et al. *Deep learning-based predictive identification of neural stem cell differentiation* (**Manuscript submitted**)
