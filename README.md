# Reptile-EEG python library
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
## Reptile-EEG is a meta-learning library for initial weights optimization of Neural Network classifiers for EEG data.
### What presented here:

- source code for library (Reptile_EEG)
- data loading example with Braindecode (dataloading_example.ipynb)
- example with full pipeline of meta-learning and evaluation (full_pipeline_example.ipynb)

## About this library

This library allows one to prepare weights for neural networks basing on data from several subjects,
in such way that these weights are most the efficient for further training.
Due to the functionality of automatic selection of meta-learning hyperparameters,
this library can be used without a deep understanding of meta-learning.

### What can this library be used for?


- reducing the computational cost of fine-tuning for new subjects
- reducing the amount of data required to achieve the sufficient classification Accuracy on fine-tuning
- obtain better classification Accuracy with the same amount of data available for training

### Citation
Citation will be added soon.