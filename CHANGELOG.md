# Changelog

## 0.2.0 (unreleased)

### Fixed
- Installing from source failed: `pyproject.toml` listed the authors in a format Poetry rejects.
- Iterative solver calls used `tol`, removed in SciPy 1.14; they now use `rtol`.
- Tests did not collect with NumPy 2 (`np.unicode_`).

### Changed
- Packaging moved to PEP 621 with hatchling. Python >=3.10, SciPy >=1.12, no upper bounds.
- The Docker image builds from the local package on `python:3.12-slim`.

### Added
- Continuous integration on Linux and macOS, Python 3.10 to 3.13.

## 0.1.1a1 (2024-10-22)

First release on PyPI.
