Bundesliga Data Shootout
==============================

A short description of the project.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


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
4. Install dependencies:
    ```sh
    poetry install
    ```

These steps have to be executed only once. To use this environment later just activate it:
```sh
conda activate dfl
```

If you want to use GPU, you have to also install [CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive).
