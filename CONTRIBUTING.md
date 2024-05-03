# Contributing to biosspheres

The following are some ways of contributing:
- Answering questions and participating.
- Fixing bugs, improving documentation, and other maintenance work.
- Triaging issues.

## Reporting issues or bugs

Ensure the bug was not already reported by searching on GitHub under Issues.

If you're unable to find an open issue addressing the problem, open a new one.
Be sure to include a title and clear description, as much relevant
information as possible, and a code sample or an executable test case
demonstrating the expected behavior that is not occurring.

## Contributing code

When submitting a pull request, we ask you to check the following:

- Unit tests, documentation, and code style are in order. It's also OK to submit
work in progress if you're unsure of what this exactly means, in which case
you'll likely be asked to make some further changes.
  
- The contributed code will be licensed under biosspheres license. If you did
not write the code yourself, you ensure the existing license is compatible
and include the license information in the contributed files, or obtain
permission from the original author to relicense the contributed code.

## Coding conventions

- For formatting the code we use the [Black code style](https://black.readthedocs.io/en/stable/the_black_code_style/index.html)
with the options `-l 80 -t py39`. Black is a PEP 8 compliant opinionated formatter with its own style.
- For the docstrings, we mainly follow the numpy format see
[numpy style guide](https://numpydoc.readthedocs.io/en/latest/format.html).
- We usually add type hints.

## Feature request

If you are interested in adding a new feature to biosspheres, consider
submitting your feature proposal to issues with the tag "feature-request"
