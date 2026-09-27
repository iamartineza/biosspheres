# Biosspheres

[![tests](https://github.com/iamartineza/biosspheres/actions/workflows/tests.yml/badge.svg)](https://github.com/iamartineza/biosspheres/actions/workflows/tests.yml)
[![docs](https://github.com/iamartineza/biosspheres/actions/workflows/docs.yml/badge.svg)](https://iamartineza.github.io/biosspheres/)
[![PyPI](https://img.shields.io/pypi/v/biosspheres)](https://pypi.org/project/biosspheres/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

A Python-based solver for Laplace and Helmholtz scattering by
multiple disjoint spheres, utilizing spherical harmonic 
decomposition and local multiple trace formulations. 

Its main routines are for:
- Computing boundary integral operators evaluated and tested
against spherical harmonics.
- Building Calderón operators.
- Solving transmission problems using the multiple trace formulation.
- Coupling the problems with ordinary differential equations in time.

# Installation

Tested with **Python 3.10 to 3.13** on Linux and macOS.

We recommend to install the package in its own python environment.

## Via pip

For the minimum installation:

`pip install biosspheres`

For the installation including the dependencies necessary for running jupyter notebooks:

`pip install "biosspheres[all]"`

## From source

```
git clone https://github.com/iamartineza/biosspheres
cd biosspheres
pip install -e ".[dev]"
pytest
```

## Docker

The biosspheres-notebook Docker image runs biosspheres with Python 3.12 and JupyterLab.
It can be built and run with:

```
docker build -t biosspheres-notebook .
docker run -v $(pwd):/root/shared -w "/root/shared" -p 8888:8888 biosspheres-notebook
```

## Additional comments

The file `requirements.txt` has the list of required package for biosspheres to work, 
they can be installed using `pip install -r requirements.txt`

# How to use

The documentation, with the API reference and the notebooks rendered, is at
https://iamartineza.github.io/biosspheres/. The notebooks are also in the folder
"notebooks".

# Contributing

Read the file `CONTRIBUTING.md` and also see the Jupyter notebooks.


# Acknowledgments

## External packages used in biosspheres

### numpy

Library for arrays, vectors and matrices in dense format, along
with routines of linear algebra, norm computation, dot product,
among others. It also computes functions like sine, cosine, exponential, etc.

- [Web page](https://numpy.org/).
- [GitHub repository](https://github.com/numpy/numpy).
- [Documentation](https://numpy.org/doc/stable/).

### scipy

- [Web page](https://scipy.org/).
- [GitHub repository](https://github.com/scipy/scipy).
- [Documentation](https://docs.scipy.org/doc/scipy/).

#### scipy.special

Library for special functions, as the spherical Bessel and
spherical Hankel functions.

#### scipy.sparse

Library for sparse arrays.

### pyshtools

All Legendre's functions are computed using the package pyshtools 
([documentation of pyshtools](https://shtools.github.io/SHTOOLS/index.html)).

- [Web page and documentation](https://shtools.github.io/SHTOOLS/).
- [GitHub repository of SHTOOLS](https://github.com/SHTOOLS/SHTOOLS).

### matplotlib

Library used in the examples for plotting.

- [Web page](https://matplotlib.org/).
- [GitHub repository](https://github.com/matplotlib/matplotlib).
- [Documentation](https://matplotlib.org/stable/users/index.html).
