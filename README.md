# Bundesliga Data Shootout

This project aims to detect football passes—including throw-ins and crosses—and challenges in original Bundesliga matches using a computer vision model. It was inspired by the kaggle competition: [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/overview).


[![Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white)](https://lightning.ai/docs/pytorch/latest/)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Project structure

```
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. 
│
├── src                <- Source code for use in this project.
│   │
│   ├── data           <- Code to process data
│   │
│   ├── training       <- Code to train models and then use trained models to make
│   │                     predictions
│   │
│   └── models         <- Pytorch models
│
├── environment.yml    <- Requirements for conda environment. Allows to easily install poetry and CUDA
|
├── README.md          <- The top-level README for developers using this project.
│
├── pyproject.toml     <- Project settings and poetry dependencies
|
└── tox.ini            <- tox file with settings for running tox
```

# Getting started

## How to install dependencies

1. Install `conda` in your system
2. Create environment:
   ```sh
   conda env create --file environment.yml
   ```
   If you already have the `dfl` environment, you have to first remove it:
   ```sh
   conda remove --name dfl --all
   ```
3. Activate environment:
   ```sh
   conda activate dfl
   ```
4. Install dependencies:
   ```sh
   poetry install
   ```

These steps have to be executed only once. To use this environment later just activate it:

```sh
conda activate dfl
```
