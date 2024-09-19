import pytest
import numpy as np
from biosspheres.utils.validation.inputs import (
    big_l_validation,
    radius_validation,
    bool_validation,
    radii_validation,
    pi_validation,
    pii_validation,
    n_validation,
    two_dimensional_array_check,
    square_array_check,
)


# Tests for big_l_validation
@pytest.mark.parametrize(
    "big_l, name",
    [
        (0, "big_l"),
        (1500, "big_l"),
        (3000, "big_l"),
        (np.int32(100), "big_l"),
    ],
)
def test_big_l_validation_valid(big_l, name):
    # Should not raise any exception
    big_l_validation(big_l, name)
    pass


@pytest.mark.parametrize(
    "big_l, name, err",
    [
        ("100", "big_l", "integer"),
        (10.5, "big_l", "integer"),
        (None, "big_l", "integer"),
        ([100], "big_l", "integer"),
    ],
)
def test_big_l_validation_type_error(big_l, name, err):
    with pytest.raises(TypeError) as exc_info:
        big_l_validation(big_l, name)
    assert str(exc_info.value).__contains__(err)
    pass


@pytest.mark.parametrize(
    "big_l, name, err",
    [
        (-1, "big_l", "negative"),
        (-100, "big_l", "negative"),
    ],
)
def test_big_l_validation_value_error(big_l, name, err):
    with pytest.raises(ValueError) as exc_info:
        big_l_validation(big_l, name)
    assert str(exc_info.value).__contains__(err)
    pass


def test_big_l_validation_warning():
    big_l = 3001
    name = "big_l"
    with pytest.warns(UserWarning) as record:
        big_l_validation(big_l, name)
    pass


# Tests for radius_validation
@pytest.mark.parametrize(
    "r, name",
    [
        (1.0, "radius"),
        (np.float64(2.5), "radius"),
    ],
)
def test_radius_validation_valid(r, name):
    # Should not raise any exception
    radius_validation(r, name)
    pass


@pytest.mark.parametrize(
    "r, name, err",
    [
        (2, "pi", "float"),
        ("1.0", "radius", "float"),
        (True, "radius", "float"),
        (None, "radius", "float"),
        ([1.0], "radius", "float"),
    ],
)
def test_radius_validation_invalid_type(r, name, err):
    with pytest.raises(TypeError) as exc_info:
        radius_validation(r, name)
    assert str(exc_info.value).__contains__(err)
    pass


@pytest.mark.parametrize(
    "r, name, error_msg",
    [
        (np.inf, "radius", "finite"),
        (-np.inf, "radius", "finite"),
        (np.nan, "radius", "finite"),
        (0.0, "radius", "positive"),
        (-1.0, "radius", "positive"),
    ],
)
def test_radius_validation_value_error(r, name, error_msg):
    with pytest.raises(ValueError) as exc_info:
        radius_validation(r, name)
    assert str(exc_info.value).__contains__(error_msg)
    pass


# Tests for bool_validation
@pytest.mark.parametrize(
    "b, name",
    [
        (True, "flag"),
        (False, "flag"),
        (np.bool_(True), "flag"),
        (np.bool_(False), "flag"),
    ],
)
def test_bool_validation_valid(b, name):
    # Should not raise any exception
    bool_validation(b, name)
    pass


@pytest.mark.parametrize(
    "b, name, err",
    [
        ("True", "flag", "bool"),
        (1, "flag", "bool"),
        (0, "flag", "bool"),
        (None, "flag", "bool"),
        ([True], "flag", "bool"),
    ],
)
def test_bool_validation_invalid_type(b, name, err):
    with pytest.raises(TypeError) as exc_info:
        bool_validation(b, name)
    assert str(exc_info.value).__contains__(err)
    pass


# Tests for radii_validation
@pytest.mark.parametrize(
    "radii, name",
    [
        (np.array([1.0, 2.5, 3.3], dtype=np.float64), "radii"),
    ],
)
def test_radii_validation_valid(radii, name):
    radii_validation(radii, name)
    pass


@pytest.mark.parametrize(
    "radii, name, err",
    [
        (np.array([1, 2, 3], dtype=np.int32), "radii", "float"),
        (np.array([1.0, 2.5, "3.3"], dtype=object), "radii", "float"),
        ("not an array", "radii", "array"),
        (123, "radii", "array"),
        (None, "radii", "array"),
        ({1: "a"}, "radii", "array"),
        ([1.0, 2.0], "radii", "array"),
    ],
)
def test_radii_validation_type_error(radii, name, err):
    with pytest.raises(TypeError) as exc_info:
        radii_validation(radii, name)
    assert str(exc_info.value).__contains__(err)
    pass


@pytest.mark.parametrize(
    "radii, name, err",
    [
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "radii",
            "dimension",
        ),
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64).reshape(-1, 1),
            "radii",
            "dimension",
        ),
        (np.array([], dtype=np.float64), "radii", "empty"),
        (np.array([1.0, -2.5, 3.3], dtype=np.float64), "radii", "positive"),
        (np.array([0.0, 2.0, 3.0], dtype=np.float64), "radii", "positive"),
        (np.array([1.0, np.inf, 3.0], dtype=np.float64), "radii", "finite"),
        (np.array([1.0, np.nan, 3.0], dtype=np.float64), "radii", "finite"),
    ],
)
def test_radii_validation_value_error(radii, name, err):
    with pytest.raises(ValueError) as exc_info:
        radii_validation(radii, name)
    assert str(exc_info.value).__contains__(err)
    pass


# Test for pi_validation
@pytest.mark.parametrize(
    "pi, name",
    [
        (1.0, "pi"),
        (np.float64(2.5), "pi"),
    ],
)
def test_pi_validation_valid(pi, name):
    # Should not raise any exception
    pi_validation(pi, name)
    pass


@pytest.mark.parametrize(
    "pi, name, err",
    [
        (2, "pi", "float"),
        ("1.0", "pi", "float"),
        (True, "pi", "float"),
        (None, "pi", "float"),
        ([1.0], "pi", "float"),
    ],
)
def test_pi_validation_type_error(pi, name, err):
    with pytest.raises(TypeError) as exc_info:
        pi_validation(pi, name)
    assert str(exc_info.value).__contains__(err)
    pass


@pytest.mark.parametrize(
    "pi, name, error_msg",
    [
        (np.inf, "pi", "finite"),
        (-np.inf, "pi", "finite"),
        (np.nan, "pi", "finite"),
        (0.0, "pi", "0"),
    ],
)
def test_pi_validation_value_error(pi, name, error_msg):
    with pytest.raises(ValueError) as exc_info:
        pi_validation(pi, name)
    assert str(exc_info.value).__contains__(error_msg)
    pass


# Tests for pii_validation
@pytest.mark.parametrize(
    "pii, name",
    [
        (np.array([1.0, 2.5, 3.3], dtype=np.float64), "pii"),
    ],
)
def test_pii_validation_valid(pii, name):
    pii_validation(pii, name)
    pass


@pytest.mark.parametrize(
    "pii, name, err",
    [
        (np.array([1, 2, 3], dtype=np.int32), "pii", "float"),
        (np.array([1.0, 2.5, "3.3"], dtype=object), "pii", "float"),
        ("not an array", "pii", "array"),
        (123, "pii", "array"),
        (None, "pii", "array"),
        ({1: "a"}, "pii", "array"),
        ([1.0, 2.0], "pii", "array"),
    ],
)
def test_pii_validation_type_errors(pii, name, err):
    with pytest.raises(TypeError) as exc_info:
        pii_validation(pii, name)
    assert str(exc_info.value).__contains__(err)
    pass


@pytest.mark.parametrize(
    "pii, name, err",
    [
        (np.array([0.0, 2.0, 3.0], dtype=np.float64), "pii", "zero"),
        (np.array([1.0, np.inf, 3.0], dtype=np.float64), "pii", "finite"),
        (np.array([1.0, np.nan, 3.0], dtype=np.float64), "pii", "finite"),
        (np.array([], dtype=np.float64), "pii", "empty"),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "pii",
            "dimension",
        ),
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64).reshape(-1, 1),
            "pii",
            "dimension",
        ),
    ],
)
def test_pii_validation_value_errors(pii, name, err):
    with pytest.raises(ValueError) as exc_info:
        pii_validation(pii, name)
    assert str(exc_info.value).__contains__(err)
    pass


# n validation
@pytest.mark.parametrize(
    "n, name",
    [
        (1, "n"),
        (150, "n"),
        (3000, "n"),
        (np.int32(100), "n"),
    ],
)
def test_n_validation_valid(n, name):
    # Should not raise any exception
    n_validation(n, name)
    pass


@pytest.mark.parametrize(
    "n, name, err",
    [
        ("100", "n", "integer"),
        (10.5, "n", "integer"),
        (None, "n", "integer"),
        ([100], "n", "integer"),
    ],
)
def test_n_validation_type_error(n, name, err):
    with pytest.raises(TypeError) as exc_info:
        n_validation(n, name)
    assert str(exc_info.value).__contains__(err)
    pass


@pytest.mark.parametrize(
    "n, name, err",
    [
        (0, "n", "positive"),
        (-100, "n", "positive"),
    ],
)
def test_n_validation_value_error(n, name, err):
    with pytest.raises(ValueError) as exc_info:
        n_validation(n, name)
    assert str(exc_info.value).__contains__(err)
    pass


# Test of function two_dimensional_array_check_valid
@pytest.mark.parametrize(
    "array",
    [
        (np.array([[1, 2, 3], [4, 5, 6]])),
        (np.array([[7, 8, 9], [10, 11, 12]])),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
    ],
)
def test_two_dimensional_array_check_valid(array):
    # This should not raise any exception
    two_dimensional_array_check(array, "array")
    pass


@pytest.mark.parametrize(
    "input_array, name, expected_exception, expected_message",
    [
        # Test cases where the input is not a numpy array
        ([[1, 2], [3, 4]], "array1", TypeError, "numpy array"),
        ([1, 2, 3, 4], "array2", TypeError, "numpy array"),
        ("not an array", "array3", TypeError, "numpy array"),
        (12345, "array4", TypeError, "numpy array"),
        (None, "array5", TypeError, "numpy array"),
        # Test cases where the input is a numpy array but not 2D
        (np.array([1, 2, 3]), "array6", ValueError, "2D"),
        (np.array(5), "array7", ValueError, "2D"),
        (np.array([[[1, 2], [3, 4]]]), "array9", ValueError, "2D"),
    ],
)
def test_two_dimensional_array_check_invalid(
    input_array, name, expected_exception, expected_message
):
    with pytest.raises(expected_exception) as exc_info:
        two_dimensional_array_check(input_array, name)
    assert str(exc_info.value).__contains__(expected_message)
    pass


# Square array checks
@pytest.mark.parametrize(
    "array,name",
    [
        (np.array([[1, 2], [3, 4]]), "2x2Matrix"),
        (np.array([[1]]), "1x1Matrix"),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "3x3Matrix"),
        (np.empty((0, 0)), "EmptyMatrix"),
        (np.array([[1.5, 2.5], [3.5, 4.5]]), "FloatMatrix"),
        (np.array([["a", "b"], ["c", "d"]]), "StringMatrix"),
    ],
)
def test_square_arrays_check_valid(array, name):
    square_array_check(array, name)
    pass


@pytest.mark.parametrize(
    "array,name,expected_message",
    [
        (np.array([[1, 2, 3], [4, 5, 6]]), "2x3Matrix", "not square"),
        (np.array([[1, 2], [3, 4], [5, 6]]), "3x2Matrix", "not square"),
        (np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), "2x4Matrix", "not square"),
        (np.array([[1]] * 3), "3x1Matrix", "not square"),
        (np.array([[1, 2], [3, 4], [5, 6]]), "3x2Matrix", "not square"),
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "4x3Matrix",
            "not square",
        ),
    ],
)
def test_square_arrays_check_invalid(array, name, expected_message):
    with pytest.raises(ValueError) as exc_info:
        square_array_check(array, name)
    assert str(exc_info.value).__contains__(expected_message)
