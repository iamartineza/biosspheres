# Contributing to biosspheres

The following are some ways of contributing:
- Answering questions and participating.
- Fixing bugs, improving documentation, and other maintenance work.
- Triaging issues.

## Reporting issues

When reporting issues please include as much detail as possible about your
operating system, python version, pyshtools version, numpy version, scipy
version and matplotlib version. Whenever possible, please
also include a brief, self-contained code example that demonstrates the problem.

## Contributing code

When submitting a pull request, we ask you to check the following:

- Unit tests, documentation, and code style are in order. It's also OK to submit
work in progress if you're unsure of what this exactly means, in which case
you'll likely be asked to make some further changes.
  - For formatting the code we use Black with the options `-l 80 -t py39`.
  - For the docstrings, we mainly follow the numpy format see
  [numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html).
  - We usually add type hints.
- The contributed code will be licensed under biosspheres license. If you did
not write the code yourself, you ensure the existing license is compatible
and include the license information in the contributed files, or obtain
permission from the original author to relicense the contributed code.

## Feature request

If you are interested in adding a new feature to biosspheres, consider
submitting your feature proposal to issues with the tag "feature-request"
