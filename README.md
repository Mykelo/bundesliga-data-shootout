# Bundesliga Data Shootout

This project aims to detect football passes—including throw-ins and crosses—and challenges in original Bundesliga matches using a computer vision model. It was inspired by the kaggle competition: [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/overview). The project is still a work in progress.

![PyTorch Badge](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff&style=flat)
[![Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white&style=flat)](https://lightning.ai/docs/pytorch/latest/)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white&style=flat)
![MLflow Badge](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=fff&style=flat)
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

# Solution

The objective of this project is to create and train a model that can accurately detect one of four specified football events - challenge, throw-in, play, or no event - using a sequence of video frames. Once the model is developed, it can be used to analyze a recording of a football match and identify the exact time when each event occurred.

## Preparing the dataset

To prepare the dataset, short clips from Bundesliga matches are extracted based on given annotations. All the code for this step is located under `src/data/`. Each clip is saved to a separate file, and all labels are saved in a CSV file. To execute the code, run the `src/data/make_dataset.py` script with the following command:

```sh
python -m src.data.make_dataset --frame-size 960 540 --window-size=32
```

This script extracts 32 frames around each event and resizes them to 960x540. It also samples clips with no events from the recordings. To view all available options, run the script with the `--help` flag.

## Training a model

The classification model was trained using the ResNet 3D architecture, and its code can be found in `src/models/models.py`. To simplify the training process on the prepared dataset, the model was wrapped with a Lightning Module. The training script can be found in `src/training_r3d.py`, and an example of its execution is shown below:

```sh
python -m src.training.train_r3d.py --max-epochs=10 --video-size 140 250 --batch-size=8
```

This command trains the model for 10 epochs, scales each frame to 140x250, and sets the batch size to 8. To see all available options, use the `--help` flag when running the script.
