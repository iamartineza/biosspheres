# Biosspheres

A Python-based solver for Laplace and acoustic scattering by multiple disjoint spheres, utilizing spherical harmonic decomposition and local multiple trace formulations.

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
dependencies. The installation
can fail depending on the machine, there are known issues, for example
      - [pyshtools issue #409 (M1-Mac)](https://github.com/SHTOOLS/SHTOOLS/issues/409#issuecomment-1749035288),
      - [pyshtools issue #327 (Arch)](https://github.com/SHTOOLS/SHTOOLS/issues/327#issuecomment-1531333702),
      - [pyshtools issue #377 (linux libtool)](https://github.com/SHTOOLS/SHTOOLS/issues/377#issuecomment-1531289262), 
      - See also the official documentation to [Install pyshtools](https://pypi.org/project/pyshtools/#installation).
- Install pip (if not installed)
`conda install pip`
- Install biosspheres
`pip install --editable .`

## Second example instructions

If the first example does not work, and it is not because of pyshtools, 
an installation using as a reference a list of packages installed in a 
successful installation could work. See the files:
- env_example_1.txt
- env_biosspheres.txt

## Docker

The biosspheres-notebook Docker image is configured to run biosspheres with Python 1.10. This can be done running:

```
docker run -v $(pwd):/root/shared -w "/root/shared" -p 8888:8888 pescapil/biosspheres-notebook
```

# Comments about the code

## About the packages used

### numpy

Library for arrays, vectors and matrices in dense format, along with routines of linear algebra, norm computation, dot product. It also computes functions like sine, cosine, exponential, etc.

- [Web page](https://numpy.org/).
- [GitHub repository](https://github.com/numpy/numpy).
- [Documentation](https://numpy.org/doc/stable/).

### scipy

- [Web page](https://scipy.org/).
- [GitHub repository](https://github.com/scipy/scipy).
- [Documentation](https://docs.scipy.org/doc/scipy/).

#### scipy.special

Library for special functions, as the spherical Bessel and spherical Hankel

#### scipy.sparse

Library for sparse arrays.

### pyshtools

All Legendre's functions are computed using the package pyshtools 
([documentation of pyshtools](https://shtools.github.io/SHTOOLS/index.html)).

Before running in a parallel see [pyshtools issue #385 (thread safety)](https://github.com/SHTOOLS/SHTOOLS/issues/385). 

- [Web page and documentation](https://shtools.github.io/SHTOOLS/).
- [GitHub repository of SHTOOLS](https://github.com/SHTOOLS/SHTOOLS).

### matplotlib

Library used in the examples for plotting.
- [Web page](https://matplotlib.org/).
- [GitHub repository](https://github.com/matplotlib/matplotlib).
- [Documentation](https://matplotlib.org/stable/users/index.html).

