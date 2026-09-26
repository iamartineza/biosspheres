biosspheres
===========

A Python-based solver for Laplace and Helmholtz scattering by multiple disjoint
spheres, utilizing spherical harmonic decomposition and local multiple trace
formulations.

Its main routines are for:

- Computing boundary integral operators evaluated and tested against spherical
  harmonics.
- Building Calderón operators.
- Solving transmission problems using the multiple trace formulation.
- Coupling the problems with ordinary differential equations in time.

.. code-block:: console

   pip install biosspheres

.. toctree::
   :maxdepth: 1
   :caption: Notebooks

   notebooks/0.biosspheres_overview
   notebooks/quadrature_overview
   notebooks/selfinteractions_laplace_overview
   notebooks/3_2_crossinteraction_laplace_overview
   notebooks/2_1_selfinteractions_helmhotlz_overview
   notebooks/2_2_crossinteraction_helmholtz_overview
   notebooks/spherearrangements_overview
   notebooks/studying_sh_expansions
   notebooks/e1.laplace_transmission_mtf
   notebooks/e2.mtf_time_coupled_Kavian
   notebooks/e3.mtf_time_coupled_e2_FitzHughNagumo
   notebooks/mtf_times

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
