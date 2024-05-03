---
title: 'biosspheres'
tags:
  - Python
  - boundary integral formulation
  - boundary integral operators
  - spectral discretization
authors:
  - name: Isabel A. Martínez-Ávila
    orcid: 0000-0002-0803-6126
    equal-contrib: true or false
    affiliation: 1
    corresponding: true # (This is how to denote the corresponding author)
  - name: Paul
    orcid: # if not delete this
    equal-contrib: true or false # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: 
    orcid: # if not delete this
    equal-contrib: true or false # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations: # examples
 - name: Institution Name, Country
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Institution Name, Country
   index: 3
date: 06 April 2024 # Change later
bibliography: paper.bib
---

# Summary

Write summary.

# Statement of need

Scattering by spheres can be efficiently solved through spherical harmonic functions. These functions diagonalize the Laplace operator for spheres in three dimensions, enabling a sparse representation of the operators under study. Typically, solvers based on spherical harmonics are restricted to solving transmission problems for a single sphere. However, there is currently no open-source library available for performing scattering problems involving multiple objects.

Biosspheres offers a comprehensive suite of routines designed to efficiently assemble boundary integral operators within spherical function spaces for a collection of three-dimensional disjoint spheres. In its operations, Biosspheres leverages the capabilities of pyshtools to accurately evaluate spherical harmonic functions, ensuring precise and reliable computations.

Direct applications serve as benchmarks for comparing and testing BEM solvers. They are also valuable for rapidly computing complex problems involving a large number of spheres. Furthermore, they can generate solutions for a range of scatterers, with direct relevance to uncertainty quantification, scientific machine learning, and cloaking.

# Mathematics

Consider the Laplace (resp., Helmholtz) scattering problem for $N$ disjoint spheres.

Be succinct with the mathematics. We could perhaps focus on Laplace or Helmholtz.

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

# References