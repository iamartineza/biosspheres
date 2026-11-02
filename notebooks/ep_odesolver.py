import numpy as np
import scipy.linalg as la
from scipy.linalg._fblas import dzasum
import scipy.sparse as sparse
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from collections.abc import Callable
import pyshtools
import pdb

import biosspheres.quadratures.sphere as quadratures
import biosspheres.utils.auxindexes as auxindexes
import biosspheres.utils.validation.inputs as valin
import biosspheres.formulations.mtf.timecouplings.linearoperators as tclo
import biosspheres.formulations.mtf.mtf as mtf
import biosspheres.laplace.selfinteractions as laplaceself
import biosspheres.formulations.massmatrices as mass
import biosspheres.miscella.forcouplings.oderesolutions as oderesolutions


# get function computing the right hand side of the equation for Z in the Kavian model
def get_odeZ_kavian_rhsfun(big_l, big_l_c, k_ep, v_rev, tau_ep, tau_res
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
# The function beta(v)    
    beta_function = oderesolutions.beta_kavian_leguebe_ea_2014_1_sphere_2d(big_l, big_l_c,
                                                            k_ep, v_rev)

    # Needed to obtain a spherical harmonic transform
    (
        quantity_theta_points,
        quantity_phi_points,
        weights,
        pre_vector,
        spherical_harmonics,
    ) = quadratures.gauss_legendre_trapezoidal_real_sh_mapping_2d(
        big_l, big_l_c
    )
    zeros = pre_vector[2, :, 0]

    # Auxiliary functions
    pesykus, p2_plus_p_plus_q, p2_plus_p_minus_q = auxindexes.pes_y_kus(big_l)
    eles, l_square_plus_l, el_square_plus_el_divided_by_two = (
        auxindexes.eles_combination(big_l)
    )
    
    def ode_rhsfun(Z : np.ndarray, v : np.ndarray
                   ) -> np.ndarray :
        
        dZ = np.empty(np.shape(Z))
        
        #pdb.set_trace()
        temp = np.sum(
            ( beta_function(v) - Z )[:, np.newaxis, np.newaxis] * spherical_harmonics,
            axis=0
        )
        #temp = (beta_function(v) - Z) @ spherical_harmonics.swapaxes(0,1)  # @ acts along next to last dim of second matrix
        temp_2 = np.maximum(temp / tau_ep, temp / tau_res)
        coefficients = pyshtools.expand.SHExpandGLQ(
            temp_2, weights, zeros, norm=4, csphase=-1, lmax_calc=big_l
        )
        dZ[p2_plus_p_plus_q] = coefficients[
            0, pesykus[:, 0], pesykus[:, 1]
        ]
        dZ[p2_plus_p_minus_q] = coefficients[
            1, pesykus[:, 0], pesykus[:, 1]
        ]
        dZ[l_square_plus_l] = coefficients[0, eles, 0]
        
        return dZ
    
    return ode_rhsfun 


# inverse of diagonal c_m matrix
def inv_c_m_sparse_array_one_sphere(
    big_l: int, radius: float, c_m: float
):
    # Input validation
    valin.big_l_validation(big_l, "big_l")
    valin.radius_validation(radius, "radius")

    j_block_diag = mass.j_block(big_l, 1/radius, azimuthal=False)
    c_m_matrix = sparse.dia_array(
        (j_block_diag / c_m, np.array([0])),
        shape=(len(j_block_diag), len(j_block_diag)),
    )
    return c_m_matrix



def ep_solveodes_one_sphere(
    big_l: int,
    radius: float,
    sigmas: np.ndarray,
    c_m: float,
    initial_time: float,
    final_time: float,
    initial_conditions: np.ndarray,
    b_phi_function,
    i_current,
    aux_ode_rhs,
):

    pi = sigmas[1] / sigmas[0]

    a_0 = laplaceself.a_0j_matrix(big_l, radius, azimuthal=False)
    a_1 = laplaceself.a_j_matrix(big_l, radius, azimuthal=False)
    mtf_matrix = mtf.mtf_1_matrix(radius, pi, a_0, a_1)
    del a_0
    del a_1
    luM, pivM = la.lu_factor(mtf_matrix)
    del mtf_matrix

    numel = (big_l+1)**2

    # I_2NxN matrix
    iden_2n_n = 2 * tclo.mtf_coupling_iden_parts_sparse_array(
        big_l, 1, np.asarray([radius])
    )
    # C_m^-1
    inv_c_m_matrix = inv_c_m_sparse_array_one_sphere(
        big_l, radius, c_m
    )
    # sigma_Nx4N matrix (first Nx2N matrix, then extend to [0 sig] : Nx4N
    sig_N_4N = tclo.mtf_coupling_sigma_times_tau_part_sparse_array(
        big_l, 1, np.asarray([radius]), sigmas, 1.0
    ).toarray()
    sig_N_4N = np.concatenate(
        (np.zeros(np.shape(sig_N_4N)), sig_N_4N), axis=1
    )

    
    # The right hand side of the ODEs for v,Z
    def ep_odefunction_one_sphere(t, state):
        v = state[0:numel]
        Z = state[numel:2*numel]
        
        b = b_phi_function(t)
        aux = iden_2n_n @ v
        b += np.concatenate((-aux,aux), axis=0)
        Jrhs = la.lu_solve((luM, pivM), b)
        Jrhs = sig_N_4N @ Jrhs
        
        dv = inv_c_m_matrix @ (-i_current(v,Z) - Jrhs)
        dZ = aux_ode_rhs(Z, v)
    
        return np.concatenate((dv, dZ), axis=0)

    # solve the problem using BDF method for stiff problems
    solution = solve_ivp(ep_odefunction_one_sphere, (initial_time, final_time), initial_conditions, method='LSODA')
    #t_eval = np.linspace(initial_time, final_time, 2**13)
    #solution = odeint(ep_odefunction_one_sphere, initial_conditions, t_eval, tfirst=True)    
    
    return solution
