import pytest
import numpy as np
from biosspheres.formulations.massmatrices import (
    j_block,
    two_j_blocks,
    n_j_blocks,
    n_two_j_blocks,
)


@pytest.mark.parametrize(
    "big_l, r, azimuthal, expected_shape",
    [
        (0, 1.0, True, (1,)),
        (0, 5.0, False, (1,)),
        (1, 1.5, True, (2,)),
        (5, 2.0, False, (36,)),
    ],
)
def test_j_block_correct_output(big_l, r, azimuthal, expected_shape):
    """
    Test that j_block returns the correct output array with expected
    shape and values.
    """
    expected = r**2 * np.ones(expected_shape[0])
    result = j_block(big_l, r, azimuthal)

    assert isinstance(result, np.ndarray), "Result should be a numpy array."
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {result.shape}"
    np.testing.assert_array_equal(
        result, expected, "j_block did not return the expected values."
    )
    pass


@pytest.mark.parametrize(
    "big_l, r, azimuthal, expected_exception, match",
    [
        (2.5, 1.0, True, TypeError, "big_l must be an integer"),
        (2, "1.0", True, TypeError, "r must be a float"),
        (5, 2.0, "True", TypeError, "azimuthal must be a boolean"),
    ],
)
def test_j_block_input_validation(
    big_l, r, azimuthal, expected_exception, match
):
    """
    Test that j_block raises appropriate exceptions for invalid inputs.
    """
    with pytest.raises(expected_exception, match=match):
        j_block(big_l, r, azimuthal)


@pytest.mark.parametrize(
    "big_l, r, azimuthal, expected_shape",
    [
        (0, 1.0, True, (2,)),
        (0, 3.0, False, (2,)),
        (1, 2.0, True, (4,)),
        (5, 3.0, False, (72,)),
    ],
)
def test_two_j_blocks_correct_output(big_l, r, azimuthal, expected_shape):
    """
    Test that two_j_blocks returns the correct output array with
    expected shape and values.
    """
    expected = r**2 * np.ones(expected_shape[0])
    result = two_j_blocks(big_l, r, azimuthal)

    assert isinstance(result, np.ndarray), "Result should be a numpy array."
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {result.shape}"
    np.testing.assert_array_equal(
        result, expected, "two_j_blocks did not return the expected" "values."
    )


@pytest.mark.parametrize(
    "big_l, r, azimuthal, expected_exception, match",
    [
        (-1, 1.0, True, ValueError, "big_l must be non-negative"),
        (2, 0.0, True, ValueError, "r must be positive"),
        (2, 1.0, "False", TypeError, "azimuthal must be a boolean"),
    ],
)
def test_two_j_blocks_input_validation(
    big_l, r, azimuthal, expected_exception, match
):
    """
    Test that two_j_blocks raises appropriate exceptions for invalid
    inputs.
    """
    with pytest.raises(expected_exception, match=match):
        two_j_blocks(big_l, r, azimuthal)


@pytest.mark.parametrize(
    "big_l, radii, azimuthal, expected_shape, expected_values",
    [
        (1, np.array([1.0, 2.0]), True, (4,), np.array([1.0, 1.0, 4.0, 4.0])),
        (2, np.array([3.0]), False, (9,), 9.0 * np.ones(9)),
        (0, np.array([1.0, 2.0, 3.0]), True, (3,), np.array([1.0, 4.0, 9.0])),
        (0, np.array([1.0, 2.0, 3.0]), False, (3,), np.array([1.0, 4.0, 9.0])),
    ],
)
def test_n_j_blocks_correct_output(
    big_l, radii, azimuthal, expected_shape, expected_values
):
    """
    Test that n_j_blocks returns the correct concatenated output array
    with expected shape and values.
    """
    result = n_j_blocks(big_l, radii, azimuthal)

    assert isinstance(result, np.ndarray), "Result should be a numpy array."
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {result.shape}"
    np.testing.assert_array_equal(
        result,
        expected_values,
        "n_j_blocks did not return the expected values.",
    )


@pytest.mark.parametrize(
    "big_l, radii, azimuthal, expected_exception, match",
    [
        (
            1,
            np.asarray([1, 2, 3], dtype=np.int32),
            True,
            TypeError,
            "radii must be an array of floats",
        ),
        (-1, np.asarray([1.0]), True, ValueError, "big_l must be non-negative"),
        (
            2,
            np.asarray([1.0, 2.0, 3.0]),
            "False",
            TypeError,
            "azimuthal must be a boolean",
        ),
    ],
)
def test_n_j_blocks_input_validation(
    big_l, radii, azimuthal, expected_exception, match
):
    """
    Test that n_j_blocks raises appropriate exceptions for invalid radii
    inputs.
    """
    with pytest.raises(expected_exception, match=match):
        n_j_blocks(big_l, radii, azimuthal)


@pytest.mark.parametrize(
    "big_l, radii, azimuthal, expected_shape, expected_values",
    [
        (1, np.array([1.0, 2.0]), True, (8,), np.array([1.0] * 4 + [4.0] * 4)),
        (2, np.array([3.0]), False, (18,), np.array([9.0] * 18)),
        (
            0,
            np.array([1.0, 2.0, 3.0]),
            True,
            (6,),
            np.repeat(np.array([1.0, 2.0, 3.0]) ** 2, 2),
        ),
        (
            0,
            np.array([1.0, 2.0, 3.0]),
            False,
            (6,),
            np.repeat(np.array([1.0, 2.0, 3.0]) ** 2, 2),
        ),
    ],
)
def test_n_two_j_blocks_correct_output(
    big_l, radii, azimuthal, expected_shape, expected_values
):
    """
    Test that n_two_j_blocks returns the correct concatenated output
    array with expected shape and values.
    """
    result = n_two_j_blocks(big_l, radii, azimuthal)

    assert isinstance(result, np.ndarray), "Result should be a numpy array."
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {result.shape}"
    np.testing.assert_array_equal(
        result,
        expected_values,
        "n_two_j_blocks did not return the expected values.",
    )


@pytest.mark.parametrize(
    "big_l, radii, azimuthal, expected_exception, match",
    [
        (
            1,
            np.asarray([1, 2, 3], dtype=np.int32),
            True,
            TypeError,
            "radii must be an array of floats",
        ),
        (-1, np.asarray([1.0]), True, ValueError, "big_l must be non-negative"),
        (
            2,
            np.asarray([1.0, 2.0, 3.0]),
            "False",
            TypeError,
            "azimuthal must be a boolean",
        ),
    ],
)
def test_n_two_j_blocks_input_validation(
    big_l, radii, azimuthal, expected_exception, match
):
    """
    Test that n_two_j_blocks raises appropriate exceptions for invalid
    radii inputs.
    """
    with pytest.raises(expected_exception, match=match):
        n_two_j_blocks(big_l, radii, azimuthal)
