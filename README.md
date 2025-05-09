# EEG-Reptile python library
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
## EEG-Reptile is a meta-learning library for initial weights optimization of Neural Network classifiers for EEG data.
### What presented here:

- source code for library (EEGReptile)
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

## Examples

Use [anaconda](https://docs.conda.io/projects/conda/en/latest/index.html) to install the environment:

`conda env create -f environment.yml`

Activate installed environment eeg-reptile:

`conda activate eeg-reptile`

Start jupyter notebook:

`jupyter notebook`

Example jupyter notebooks:

- `dataloading_example.ipynb` - example of uploading data from MOABB to back-end dataset for EEG-Reptile.

  *(run first to create the dataset for the next example)*
- `full_pipeline_example.ipynb` - example of meta-learning experiment with description for main functions 

## Citation
For referencing to the EEG-Reptile library, please cite our [paper](https://arxiv.org/abs/2412.19725):

    Berdyshev, D.A., Grachev, A.M., Shishkin, S.L. and Kozyrskiy, B.L., 2024. EEG-Reptile: An Automatized Reptile-Based Meta-Learning Library for BCIs. arXiv preprint arXiv:2412.19725.
Bibtex version:

    @article{berdyshev2024eeg,
      title={EEG-Reptile: An Automatized Reptile-Based Meta-Learning Library for BCIs},
      author={Berdyshev, Daniil A and Grachev, Artem M and Shishkin, Sergei L and Kozyrskiy, Bogdan L},
      journal={arXiv preprint arXiv:2412.19725},
      year={2024}
    }
