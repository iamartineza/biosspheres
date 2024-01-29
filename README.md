# Python requirement

python>=3.9

It has been tested with python=3.9

# Installing instructions

An example installation using conda:
- Clone the repository.
- Create a new conda environment and activate it.
- Add conda-forge channel
`conda config --add channels conda-forge`
- cd to local directory.
- Install the packages in requirements.txt using
`conda install --file requirements.txt`
- If the jupyter notebooks are needed install also
`conda install --file requirements_notebooks.txt`
- Install pip (if not installed)
`conda install pip`
- Install biosspheres
`pip install --editable .`


## Comments about the code.

All Legendre's functions are computed using the package pyshtools 
([documentation of pyshtools](https://shtools.github.io/SHTOOLS/index.html))
