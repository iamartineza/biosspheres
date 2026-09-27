# Changelog

## 0.2.0 (unreleased)

### Fixed
- Installing from source failed: `pyproject.toml` listed the authors in a format Poetry rejects.
- Iterative solver calls used `tol`, removed in SciPy 1.14; they now use `rtol`.
- Tests did not collect with NumPy 2 (`np.unicode_`).
- `b_vector_1_sphere_mtf` validated `pi_inv` as an array, so every call with a float raised a `TypeError`; this broke the one-sphere solver template (#14).
- `a_0j_linear_operator.rmatvec` had an extra factor `r`; it only matched `matrix.T` for `r = 1` (#11).
- Typos in the notebooks and in the Laplace docstrings (#12).
- `rf_helmholtz_n_spheres` raised `NameError` on every call; it is rewritten, vectorized over points, in the complex spherical harmonic basis of the Helmholtz routines (#18).
- Overlapping, touching or nested (e.g. concentric) spheres gave wrong cross interactions without any error; they now raise `ValueError` (#17).

### Changed
- Packaging moved to PEP 621 with hatchling. Python >=3.10, SciPy >=1.12, no upper bounds.
- The Docker image builds from the local package on `python:3.12-slim`.

### Added
- Continuous integration on Linux and macOS, Python 3.10 to 3.13, and for the notebooks (#15).
- Tests for the Laplace and Helmholtz operators and the MTF, converted from the former check scripts, with closed-form references for one sphere (#11, #13, #14).
- Documentation built with Sphinx and published on GitHub Pages (#12).
- Tests against the Mie series for one sphere, for the traces and for the field in and out of the sphere (#18, #20).

## 0.1.1a1 (2024-10-22)

First release on PyPI.
