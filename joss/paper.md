---
title: 'biosspheres: boundary integral operators and multiple traces formulations on ensembles of spheres'
tags:
  - Python
  - boundary integral equations
  - multiple traces formulation
  - spherical harmonics
  - electropermeabilization
authors:
  - name: Isabel A. Martínez-Ávila
    orcid: 0000-0002-0803-6126
    corresponding: true
    affiliation: 1
  - name: Carlos Jerez-Hanckes
    orcid: 0000-0001-8225-9558
    affiliation: "2, 3"
  - name: Paul Escapil-Inchauspé
    orcid: 0000-0002-2187-9232
    affiliation: 4
  - name: Tobias Gebäck
    orcid: 0000-0001-9899-9366
    affiliation: 5
affiliations:
  - name: Universidad Técnica Federico Santa María, Chile
    index: 1
  - name: KTH Royal Institute of Technology, Stockholm, Sweden
    index: 2
  - name: Inria Chile, Santiago, Chile
    index: 3
  - name: Facultad de Ingeniería y Ciencias, Universidad Adolfo Ibáñez, Santiago, Chile
    index: 4
  - name: Chalmers University of Technology and University of Gothenburg, Sweden
    index: 5
date: 27 September 2026
bibliography: paper.bib
---

# Summary

`biosspheres` is a Python package for boundary integral equations posed on
ensembles of disjoint spheres in three dimensions. It assembles the Laplace and
Helmholtz single layer, double layer, adjoint double layer and hypersingular
operators in a basis of real or complex spherical harmonics, both on each
sphere (self-interactions) and between pairs of spheres (cross-interactions).
With these operators it builds the Calderón operators and the local multiple
traces formulation (MTF) of transmission problems [@Hiptmair2011], solves them
with direct or iterative solvers, and reconstructs the volume solution from the
computed traces. The same machinery is coupled in time with nonlinear ordinary
differential equations on the sphere boundaries, which is the model used for
biological cells under electrical stimulation [@Henriquez2016; @Henriquez2018;
@MartinezAvila2024]. Membrane models of @Kavian2014 and of FitzHugh–Nagumo
type [@FitzHugh1961] are included.

# Statement of need

On a sphere, spherical harmonics diagonalize the boundary integral operators of
the Laplace and Helmholtz equations. A Galerkin discretization with
spherical harmonics is then spectrally accurate, needs no mesh, and turns the
self-interaction blocks into diagonal matrices with closed-form entries. When
several spheres are present the cross-interaction blocks are dense but still
smooth, and they can be computed to near machine precision with modest
quadrature. This makes spheres the natural setting for two kinds of work.

The first is modelling. Cells in suspension or in tissue are often represented
as spheres surrounded by a thin membrane, and the transmembrane potential obeys
an ODE driven by the jump of the electric potential across the membrane. The MTF
couples one pair of traces per sphere with the exterior medium, which keeps
each subdomain independent and makes the time coupling explicit
[@Henriquez2018; @MartinezAvila2024]. `biosspheres` provides this coupling for
arbitrary numbers of spheres and conductivities.

The second is verification. For one sphere the discrete MTF is diagonal in
each degree and reproduces the Mie series of the transmission problem exactly,
up to the truncation degree; the package tests this against the Mie coefficients
to $10^{-14}$. For several spheres the solutions converge exponentially in the
maximum degree $L$. They therefore serve as analytic or near-analytic reference
solutions for general boundary element codes, for preconditioners of the MTF
[@Ayala2022; @EscapilInchauspe2025], and for generating families of solutions
for uncertainty quantification or scientific machine learning.

The package is aimed at researchers in numerical analysis of boundary integral
equations and at modellers of cell electrophysiology who need fast, accurate
solutions on sphere ensembles.

# State of the field

Multiple scattering by spheres has a long history in electromagnetics, where
T-matrix codes are the standard: MSTM [@Mackowski2011], CELES [@Egel2017],
Smuthi [@Egel2021] and treams [@Beutel2024]. For acoustics, multipole
reexpansion methods [@Gumerov2002] and packages such as MultipleScattering.jl
[@MultipleScatteringjl] and biem-helmholtz-sphere [@biemhelmholtzsphere] solve
Helmholtz scattering by several spheres. General boundary element libraries
such as Bempp-cl [@Betcke2021] handle arbitrary geometries, including spheres,
through surface meshes.

None of these covers the setting `biosspheres` was written for. T-matrix codes
work with the Maxwell equations or with Helmholtz scattering by impenetrable or
homogeneous particles, and do not expose the boundary integral operators
themselves. They do not treat the Laplace equation, which is the relevant model
for quasi-static electrical stimulation of cells. Mesh-based boundary element
libraries give only algebraic convergence on spheres, and a spectral basis
cannot be added to them without replacing their assembly. `biosspheres`
therefore implements the operators directly in the spherical harmonic basis,
with the MTF and the time coupling on top, and relies on SHTools
[@Wieczorek2018] for the associated Legendre functions and on NumPy and SciPy
[@2020NumPy-Array; @2020SciPy-NMeth] for linear algebra and special functions.

# Software design

The package is organized by equation and by level of abstraction. The modules
`laplace` and `helmholtz` contain the operators; `formulations` builds mass
matrices, the MTF systems and their right-hand sides; `quadratures` implements
the Gauss–Legendre and trapezoidal rules on the sphere and the change of
coordinates between spheres; `miscella` holds harmonic expansions of point
sources and plane waves, membrane current models and sphere arrangements.

Self-interactions are returned as the diagonals of the operators, with an
azimuthal variant that keeps only order-zero harmonics when the problem is
axisymmetric. Cross-interactions between spheres $j$ and $s$ are computed
semi-analytically: the action of an operator of sphere $j$ on a spherical
harmonic of sphere $j$ is known in closed form outside sphere $j$, and it is
tested against the spherical harmonics of sphere $s$ with a quadrature of degree
$L_c \geq L$ on sphere $s$. Two implementations are provided for each block. The
first stores the quadrature points in one-dimensional arrays and is simple to
read; the second uses two-dimensional arrays and a faster spherical harmonic
transform. The tests check that both agree. The double layer, adjoint double
layer and hypersingular blocks are obtained from the single layer block through
their algebraic relations in this basis, instead of being integrated
separately, which reduces the cost of assembling the Calderón operator.

Each MTF system is available as a dense matrix and as a
`scipy.sparse.linalg.LinearOperator`. The dense form is convenient for direct
solvers and for studying spectra; the operator form avoids storing the
diagonal and sparse blocks explicitly and is the one used with GMRES. For eight
spheres with $L = 15$, the iterative solver is about three times faster than the
direct one in the timing notebook shipped with the package. A reduced MTF
eliminates the interior traces through a Schur complement; since the interior
blocks are diagonal this is exact and halves the number of unknowns.

The time-dependent problems use a semi-implicit scheme: the linear MTF system
is treated implicitly, the nonlinear membrane current explicitly, so a single
factorization of the system matrix is reused at every time step
[@MartinezAvila2024].

All public routines validate their inputs and raise explicit errors. The tests
compare the one-sphere Helmholtz operators with their closed forms in spherical
Bessel and Hankel functions and with a direct quadrature of the kernel, check
the discrete Calderón identity $(2A)^2 = I$ for the Laplace operators, and check
the
agreement between the matrix and linear-operator versions, between the one- and
two-dimensional versions, between the reduced and full MTF, in
phantom-sphere experiments whose exact solution is known, and between the
one-sphere Helmholtz MTF and the Mie series. Jupyter notebooks
document each module and run in continuous integration together with the tests.

![(a) Real part of the total field on the plane $y = 0$ for a plane wave of
wave number $k_0 = 4$ scattered by 27 spheres of refractive index 2 with random
radii and positions; the arrow shows the direction of incidence. (b) Relative
$\ell^2$ difference of the field on that plane with respect to $L = 14$: the
error decreases by an order of magnitude every two degrees.\label{fig:spheres}](figure.png)

# Research impact statement

The numerical experiments of @MartinezAvila2024, which follow the
electropermeabilization of up to eight cells in a cubic lattice under nonlinear
membrane dynamics, were computed with the code that became `biosspheres`. The
simulation of Section 4.3.2 of that article is shipped as a notebook that runs
in continuous integration, so it stays reproducible as the package evolves.

FAIR-SciML [@fairsciml] uses `biosspheres` to generate datasets for neural
operators: each sample is the field scattered by a random array of 27 penetrable
spheres (\autoref{fig:spheres}), computed in about two minutes on one core with
$L = 8$, at which the field is within $2 \times 10^{-6}$ of the $L = 14$
solution. A DeepONet trained on a first set of 1536 samples reaches a relative
$\ell^2$ test error of 0.30, down from 0.38 with 512 samples. This use depends
on the analytic character of the solutions: the labels are exact up to the
truncation degree, and a dataset of thousands of multi-sphere configurations
would be costly to produce, and harder to certify, with a mesh-based solver.

The same property makes `biosspheres` a verification tool beyond its own
applications. @EscapilInchauspe2025 validates a multiple traces solver against
the Mie series on a single sphere; for the Laplace and Helmholtz equations,
`biosspheres` extends such references to any number of penetrable spheres, with
the one-sphere case matching the Mie series to $10^{-14}$ and exponential
convergence in $L$ otherwise. The package is installable from PyPI, has an API
reference and tutorial notebooks, and is tested on Linux and macOS with Python
3.10 to 3.13.

# AI usage disclosure

# Acknowledgements

# References
