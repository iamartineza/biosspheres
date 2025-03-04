# Biosspheres

A Python-based solver for Laplace and Helmholtz scattering by
multiple disjoint spheres, utilizing spherical harmonic
decomposition and local multiple trace formulations.

Its main routines are for:
- Computing boundary integral operators evaluated and tested
against spherical harmonics.
- Building Calderón operators.
- Solving transmission problems using the multiple trace formulation.

# Installation

Tested with **python 3.9**

We recommend to install the package in its own python environment.

## Via pip

The package is available in PyPi.

For the minimum installation:

`pip install biosspheres`

For the installation including the dependencies necessary for running jupyter notebooks:

`pip install biosspheres[all]`

## Docker

The biosspheres-notebook Docker image is configured to run biosspheres with Python 3.10. This can be done running:

```
docker run -v $(pwd):/root/shared -w "/root/shared" -p 8888:8888 pescapil/biosspheres-notebook
```

## Additional comments

The file `requirements.txt` has the list of required package for biosspheres to work, 
they can be installed using `pip install -r requirements.txt`

# How to use

See the Jupyter notebook examples in the folder "notebooks" to see how biosspheres can be used.

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
