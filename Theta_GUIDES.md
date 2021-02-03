# Guides on installation

This documentation provides guides on how to install PETSc and torchdiffeq on Argonne's Theta supercomputer.

---
## Prerequisite on Theta
```
module add cudatoolkit
module add cray-python
```
For convenience, these commands can be added into your ~/.bash_profile script so they will be added automatically at login.

## Installation of PETSc
```
git clone https://gitlab.com/petsc/petsc.git
cd petsc
git checkout hongzh/fix-veccreate-cuda
./configure --download-revolve --with-cuda --with-fc=0 --with-debugging=no --with-batch --with-petsc4py PETSC_ARCH=arch-theta-cuda-opt
```
Follow the printed instructions at the end of configure to do a `make`.

## Installation of torchdiffeq
```
git clone https://github.com/rtqichen/torchdiffeq.git
cd torchdiffeq
pip install -e .
```
