[![Documentation Status](https://readthedocs.org/projects/tiscla/badge/?version=latest)](https://tiscla.readthedocs.io/en/latest/?badge=latest)

# TISCLA1: Tissue Classifier

This tissue classifier is trained on gene expression data from GTEx and can be used to initialize weights or as an architecture for training platform/lab-specific classifiers.

## Contents
- [Running apps](#running-apps)
- [Repo Structure](#repo-structure)
- [Getting started](#getting-started)
  - [Install libraries](#install-libraries)
  - [Conda environment set-up](#conda-environment-set-up)
  - [Start notebook server](#start-notebook-server)

## Running apps

Applications should be run from `src/`, relative to the `src/sbr` modules.

After setting up the environment and starting the notebook server, apps can be either:
1. run from the commandline
2. run from within a notebook
3. constructed from the functions available through the `sbr` package

Trained model, weights, and class names are provided under `saved_models`, as noted below.

### Tissue Classifier

#### Classify a sample
To classify a sample, format and load it, load a model and class names, and call model.predict as below.

```
# load the source-controlled model
model=load_model(  '../saved_models/gtex_model.h5')
# load in the class_names to decode the 1-hot vector
with open("../saved_models/class_names.txt") as f:
    class_names = f.read().splitlines()
# load in some data from disk with multicategorical_split or byod (suggest log2 normalized)
class_names[np.argmax(model.predict(my_sample))]
```
If using samples from `multicatigorical_split`, use samples in form: `x_test[i:i+1]`


#### Train the model
The model can be trained using the notebook `Training_pipeline.ipynb`, or by running the following program:
```
python gtex-train.py
```

#### Get the data
To retrieve and preprocess gtex data into a format the model can use:
```
./get_gtex.sh
python gtex-preprocess.py
```

## Repo Structure

This repository is structured as follows:
 - `src/`:  **Notebooks and applications** in the root, and all relevant source code under `sbr`
 - `src/sbr`:  useful functions that are sourced by the notebooks
 - `src/data`:  created automatically by the apps, stores output from the classifier model
 - `saved_models`:  weights saved from models trained by apps under `src`. These can be loaded and used immediately for predictions without training if the user has knowledge of how to do that.
 - `tiscla_env.yml`: a tested conda environment; note, `LD_LIBRARY_PATH` variable must also be **exported**, see below

## Getting started

### Install libraries

There are a few different approaches to running this code. One is to
create a Google colab notebook and import the data/code there. Another
is to set up docker and run a GPU-enabled container. Below are
instructions for a third option, set up a python environment. These
instructions have been tested on Ubuntu.

To get started, first install the required libraries inside a virtual environment:

```
# install nvidia drivers if you haven't already:
#https://www.nvidia.com/Download/index.aspx

# make a tensorflow environment that works with
# 11th Gen Intel(R) Core(TM) i9-11900H @ 2.50GHz   2.50 GHz
# check your card:
nvidia-smi --query-gpu=gpu_name --format=csv|tail -n 1
# NVIDIA GeForce RTX 3050 Ti Laptop GPU


# install mamba for faster package management:
# sometimes you have to repeat a mamba command, its still faster than conda
conda install -n base conda-forge::mamba
```
###  Conda environment set-up

Create a virtual environment using mamba (because it's faster than conda).

```
mamba create -n classifier tensorflow-gpu -c conda-forge
conda activate classifier
# tested under:
# python v3.9.13, tensorflow v 2.6.2; you'll lget the right version of libcusolver.so (v11) for cudatoolkit=11.7
# 
# CAVEAT: tensorflow libraries and nvidia drivers have some strange inter-dependencies and bugs that make package management dodgey, at best, especially in earlier chipsets. In my experience, these are becoming more stable with evolving tensorflow/keras combinations, and using the latest hardware/drivers/libs, suchas offered by Google colab, go a long way towards solving this problem. However, the following notes include hints for overcoming problems that may be caused by GPU chipsets/drivers that may not be the most current.

pip install tf-explain

# #############
# BEGIN: UPGRADE (or downgrade) TF:
# If you wantt o upgrade tf, do at your own risk:
# install/uninstall/install some tf/cuda packages to get the right version combination

# uninstall tensorflow so it's linked to the version your cudakit needs later on
pip uninstall tensorflow

# xxx this leaves behind libcusolver11.so.11 somehow?
conda uninstall cudatoolkit # mamba didn't work for me

# install tool to query your nvidia toolkit version
mamba install cuda-nvcc -c nvidia
nvcc --version

# Assuming nvcc version is 11.7: will bring correct cudnn, and libcusolver.so.11:
# WARNING: 2.5.0 has a broken libcusolver
#   If you install cudatoolkit any other way for 2.5.0, libcusolver.so.10 will be installed when you need so.11, and you'll get errors
mamba install cudatoolkit=11.7 

# IF your cuda drivers need an explicit tf version, uncomment and modify the code below:
#TENSORFLOW_VERSION="==2.3" 
# ELSE use htis:
TENSORFLOW_VERSION="" # use this to explicitly set version if required by your cuda drivers
pip install tensorflow-gpu${TENSORFLOW_VERSION}
# ensure you still have the right version of libcusolver
#ls ~/miniconda3/envs/classifier/lib/libcusolver*
# or 
ls $CONDA_PREFIX/lib/libcusolver*
# END: UPGRADE (or downgrade) TF
# ######################

# make sure tf can find your cuda libraries
pushd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
echo -e "#\!/bin/sh\nexport SAVE_PREVIOUS_LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}\nexport LD_LIBRARY_PATH=${CONDA_PREFIX}/lib" > ./etc/conda/activate.d/env_vars.sh
echo -e "#\!/bin/sh\nexport LD_LIBRARY_PATH=\${SAVE_PREVIOUS_LD_LIBRARY_PATH}"
popd
# or simply:
# conda env config vars set LD_LIBRARY_PATH=~/miniconda3/envs/classifier/lib

# pick up the environmental variable you just set
conda deactivate
conda activate classifier

mamba install --file requirements.txt -c conda-forge -c esri
# add in latest from keras-contrib to pick up new InstanceNormalization layer
pip install git+https://www.github.com/keras-team/keras-contrib.git

### test:
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
# ignore NUMA node warnings, they're harmless, see: https://forums.developer.nvidia.com/t/numa-error-running-tensorflow-on-jetson-tx2/56119/2
# I think this happens if GPU is number '0'
# All the libs should load
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# ignore NUMA node warning, see above

# CAVEAT:
# There's a weird bug in tensorflow package management that causes keras to be installed twice; 
# It might be from installation of keras-applications in requirements.txt, not sure
# if the tests above fail, try pip uninstall'ing keras by uncomment'ing below; and they should work again:
#pip uninstall keras

# save your environment so you can version control and/or use it on other machines, if desired
conda env export --file classifier_env.yml
# CAVEAT - LD_LIBRARY_PATH must be exported for tf to work; but the yml above only 'sets' it.
```

### Start notebook server
```
jupyter notebook
```
You can monitor your nvidia process with the following:
```
nvidia-smi dmon
```

