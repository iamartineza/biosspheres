# Python requirement

python==3.9

It has been tested only with python 3.9

# Installing instructions

An example installation using conda:
- Clone the repository.
- Create a new conda environment and activate it.
- Add conda-forge channel
`conda config --add channels conda-forge`
- cd to local directory.
- Install the packages in requirements.txt using
`conda install --file requirements.txt`
    - This should install the package `pyshtools==4.10.4` along with its 
dependencies (latest version in conda-forge at the moment). The installation
can fail depending on the machine, there are known issues, for example
      - [pyshtools issue #409 (M1-Mac)](https://github.com/SHTOOLS/SHTOOLS/issues/409#issuecomment-1749035288),
      - [pyshtools issue #327 (Arch)](https://github.com/SHTOOLS/SHTOOLS/issues/327#issuecomment-1531333702),
      - [pyshtools issue #377 (linux libtool)](https://github.com/SHTOOLS/SHTOOLS/issues/377#issuecomment-1531289262), 
      - See also the official documentation to [Install pyshtools](https://pypi.org/project/pyshtools/#installation).
- If the jupyter notebooks are needed install also
`conda install --file requirements_notebooks.txt`
- Install pip (if not installed)
`conda install pip`
- Install biosspheres
`pip install --editable .`


## Comments about the code.

All Legendre's functions are computed using the package pyshtools 
([documentation of pyshtools](https://shtools.github.io/SHTOOLS/index.html)).

Before running in a parallel see [pyshtools issue #385 (thread safety)](https://github.com/SHTOOLS/SHTOOLS/issues/385). 
