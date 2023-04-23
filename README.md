# Bundesliga Data Shootout

This project aims to detect football passes—including throw-ins and crosses—and challenges in original Bundesliga matches using a computer vision model. It was inspired by the kaggle competition: [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/overview).

[Lightning](https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white)

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
