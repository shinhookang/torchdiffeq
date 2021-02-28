# Guides on installation

This documentation provides guides on how to install PETSc and torchdiffeq on the GPU nodes of Argonne's Theta supercomputer. The easist way is to install them directly on the GPU nodes instead of the front login nodes.

---
## Prerequisite on ThetaGPU

1. Request an interactive node

Once you login to Theta, do the following
```
ssh thetagpusn1
qsub -n 1 -t <NODE_TIME_IN_MINUTES> -A <ALLOCATION_NAME> -I
```

2. Enable python environment:
```
source /lus/theta-fs0/software/thetagpu/conda/pt_master/2020-11-25/mconda3/setup.sh
```
Python is not avaible on thetagpusn nodes and GPU nodes. This will setup a conda environment with a recent "from scratch" build of the PyTorch repository on the master branch. 

3. Set up internet connection:
```
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
```

Now you can install python packages with `pip install`, e.g.
```
pip install Cython matplotlib scipy
```
More info can be found at:
https://www.alcf.anl.gov/support-center/theta-gpu-nodes/running-pytorch-conda

## Installation of PETSc
```
git clone https://gitlab.com/petsc/petsc.git
cd petsc
./configure --download-revolve --with-shared-libraries COPTFLAGS=-O3 CXXOPTFLAGS=-O3 FOPTFLAGS=-O3 PETSC_ARCH=arch-theta-gpu-opt --with-cuda-gencodearch=80 --with-cuda=1 --with-cudac=nvcc --with-petsc4py --download-fblaslapack
```
Follow the printed instructions at the end of configure to do a `make` and add <path_to_petsc>/petsc/arch-theta-cuda-opt/lib to PYTHONPATH.
For example, the following can be run directly or added into bash_profile script.
```
export PYTHONPATH=<path_to_petsc>/petsc/arch-theta-cuda-opt/lib:PYTHONPATH
```

## Installation of torchdiffeq
```
git clone https://github.com/rtqichen/torchdiffeq.git
cd torchdiffeq
pip install --user -e .
```
