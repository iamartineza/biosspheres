"""
Tests for biosspheres.laplace.selfinteractions: linear operators against
matrices, azimuthal against general versions, and the Calderón
projector identity for one sphere.
"""

import pytest
import numpy as np
import scipy.sparse.linalg
import biosspheres.laplace.selfinteractions as selfinteractions
import biosspheres.miscella.extensions as extensions
import biosspheres.formulations.massmatrices as mass

BUILDERS = [
    (selfinteractions.a_0j_linear_operator, selfinteractions.a_0j_matrix),
    (selfinteractions.a_j_linear_operator, selfinteractions.a_j_matrix),
]
BUILDER_IDS = ["a_0j", "a_j"]
SIZES = [(0, 1.0), (5, 2.5), (12, 0.7)]


def relative_error(approximation: np.ndarray, reference: np.ndarray) -> float:
    """
    Returns the max-norm of the difference relative to the max-norm of
    the reference.
    """
    return np.max(np.abs(approximation - reference)) / np.max(np.abs(reference))


def extend(big_l: int, v: np.ndarray) -> np.ndarray:
    """
    Extends by zero both halves of an azimuthal vector to all orders.
    """
    num = big_l + 1
    return np.concatenate(
        (
            extensions.azimuthal_trace_to_general_with_zeros(big_l, v[:num]),
            extensions.azimuthal_trace_to_general_with_zeros(big_l, v[num:]),
        )
    )


def gmres_solve(
    linear_operator: scipy.sparse.linalg.LinearOperator, b: np.ndarray
) -> np.ndarray:
    """
    Solves with GMRES at a relative tolerance of 1e-13.
    """
    solution, info = scipy.sparse.linalg.gmres(
        linear_operator, b, rtol=10 ** (-13), restart=len(b)
    )
    assert info == 0
    return solution


@pytest.mark.parametrize("builders", BUILDERS, ids=BUILDER_IDS)
@pytest.mark.parametrize("azimuthal", [True, False])
@pytest.mark.parametrize("big_l, r", SIZES)
def test_linear_operator_matches_matrix(builders, azimuthal, big_l, r) -> None:
    """
    The linear operator and the matrix of A_{j,j}^0 and A_{j,j} give the
    same products, transposed products and solutions.
    """
    operator_builder, matrix_builder = builders
    linear_operator = operator_builder(big_l, r, azimuthal)
    matrix = matrix_builder(big_l, r, azimuthal)
    b = np.random.default_rng(0).random(matrix.shape[0])
    assert relative_error(linear_operator.matvec(b), matrix @ b) < 2e-14
    assert relative_error(linear_operator.rmatvec(b), matrix.T @ b) < 2e-14
    assert (
        relative_error(
            gmres_solve(linear_operator, b), np.linalg.solve(matrix, b)
        )
        < 1e-12
    )
    pass


@pytest.mark.parametrize("builders", BUILDERS, ids=BUILDER_IDS)
@pytest.mark.parametrize("big_l, r", SIZES)
def test_azimuthal_matches_general(builders, big_l, r) -> None:
    """
    For an azimuthal right hand side extended by zero, the azimuthal and
    the general versions give the same products and solutions.
    """
    operator_builder, matrix_builder = builders
    b = np.random.default_rng(1).random(2 * (big_l + 1))
    b_general = extend(big_l, b)
    product = extend(big_l, matrix_builder(big_l, r, True) @ b)
    product_general = matrix_builder(big_l, r, False) @ b_general
    assert relative_error(product_general, product) < 2e-15
    solution = extend(big_l, gmres_solve(operator_builder(big_l, r, True), b))
    solution_general = gmres_solve(operator_builder(big_l, r, False), b_general)
    assert relative_error(solution_general, solution) < 6e-14
    pass


@pytest.mark.parametrize(
    "matrix_builder", [b[1] for b in BUILDERS], ids=BUILDER_IDS
)
@pytest.mark.parametrize("azimuthal", [True, False])
@pytest.mark.parametrize("big_l, r", SIZES)
def test_calderon_projector(matrix_builder, azimuthal, big_l, r) -> None:
    """
    With A the Galerkin matrix of the Calderón operator and M the mass
    matrix, the operator M^{-1} A satisfies (2 M^{-1} A)^2 = I, that is,
    4 A M^{-1} A = M.
    """
    matrix = matrix_builder(big_l, r, azimuthal)
    mass_matrix = mass.two_j_blocks(big_l, r, azimuthal)
    square = 4.0 * matrix @ (matrix / mass_matrix[:, np.newaxis])
    assert relative_error(square, np.diag(mass_matrix)) < 3e-14
    pass


@pytest.mark.parametrize("azimuthal", [True, False])
@pytest.mark.parametrize("big_l", [0, 4])
def test_calderon_projector_n_spheres(azimuthal, big_l) -> None:
    """
    The sparse block matrices for several spheres satisfy
    4 A M^{-1} A = M, with M the mass matrix of all the spheres.
    """
    radii = np.asarray([0.5, 1.0, 3.0])
    mass_matrix = mass.n_two_j_blocks(big_l, radii, azimuthal)
    for matrix in selfinteractions.a_0_a_n_sparse_matrices(
        len(radii), big_l, radii, azimuthal
    ):
        matrix = matrix.toarray()
        square = 4.0 * matrix @ (matrix / mass_matrix[:, np.newaxis])
        assert relative_error(square, np.diag(mass_matrix)) < 3e-14
        pass
    pass
