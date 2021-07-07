#!/bin/bash

# The python version installed in the conda setup
WGET="wget --tries=3"
# Both Miniconda2/3 can install any Python versions
CONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# install conda
test -f miniconda.sh || ${WGET} ${CONDA_URL} -O miniconda.sh
test -d ${PWD}/conda || bash miniconda.sh -b -p ${PWD}/conda
. conda/bin/activate
# clone package
conda env create -f ./conda_env.yml
conda activate avsd
#install spacy language model. Make sure you activated the conda environment
python -m spacy download en
