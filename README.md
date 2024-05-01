# Biosspheres

A Python-based solver for Laplace and acoustic scattering by
multiple disjoint spheres, utilizing spherical harmonic
decomposition and local multiple trace formulations.

Its main routines are for:
- Computing boundary integral operators evaluated and tested
against spherical harmonics.
- Building Calderón operators.
- Solving transmission problems using the multiple trace formulation. 

Tested with **python 3.9**

# Installation

## Via pip/conda

An example installation using conda:
- Clone the repository.
- Create a new conda environment (with python=3.9) and activate it.
- Add conda-forge channel if not added
`conda config --add channels conda-forge`
- cd to local directory.
- Install the packages in requirements.txt using
`conda install --file requirements.txt`
    - This should install the package `pyshtools==4.10.4` along with its 
dependencies. See its official documentation if not installed correctly
([Install pyshtools](https://pypi.org/project/pyshtools/#installation)).
- Install pip (if not installed)
`conda install pip`
- Install biosspheres
`pip install --editable .`

## Second example instructions

If the first example does not work, 
an installation using as a reference a list of packages installed in a 
successful installation could work. See the files:
- env_example_1.txt
- env_biosspheres.txt

## Docker

The biosspheres-notebook Docker image is configured to run biosspheres with Python 3.10. This can be done running:

```
docker run -v $(pwd):/root/shared -w "/root/shared" -p 8888:8888 pescapil/biosspheres-notebook
```

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
