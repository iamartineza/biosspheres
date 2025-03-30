import pytest
import numpy as np
from scipy import sparse
from biosspheres.utils.validation.inputs import (
    big_l_validation,
    float_validation,
    radius_validation,
    bool_validation,
    numpy_array_validation,
    finite_values_in_array,
    float_array_validation,
    radii_validation,
    pi_validation,
    pii_validation,
    n_validation,
    two_dimensional_array_check,
    square_array_check,
    same_shape_check,
    is_scipy_linear_op,
    same_type_check,
    is_scipy_sparse_array,
)
import biosspheres.utils.validation.inputs as valin


########################################################################
# Tests for valin.integer_validation
########################################################################
@pytest.mark.parametrize(
    "integer, name",
    [
        (0, "0"),
        (1500, "1500"),
        (-1, "-1"),
        (np.int32(100), "numpy integer"),
    ],
)
def test_integer_validation(integer, name):
    # Should not raise any exception
    valin.integer_validation(integer, name)
    pass


########################################################################
# Tests for big_l_validation
########################################################################
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


########################################################################
# n validation
########################################################################
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


########################################################################
# Test for float_validation
########################################################################
# Tests for valid floating point values
@pytest.mark.parametrize(
    "fl, name",
    [
        (0.0, "zero"),
        (1.0, "one"),
        (-1.0, "negative_one"),
        (3.14159, "pi"),
        (1e-10, "small_value"),
        (1e10, "large_value"),
        (np.float32(2.5), "numpy_float32"),
        (np.float64(3.7), "numpy_float64"),
    ],
)
def test_float_validation_valid(fl, name):
    # Should not raise any exception
    float_validation(fl, name)


# Tests for invalid types (not float)
@pytest.mark.parametrize(
    "fl, name, err",
    [
        (1, "integer", "float"),
        ("1.0", "string", "float"),
        (True, "boolean", "float"),
        (None, "none", "float"),
        ([1.0], "list", "float"),
        (np.array([1.0]), "numpy_array", "float"),
        ((1.0,), "tuple", "float"),
        ({1.0}, "set", "float"),
        ({"key": 1.0}, "dict", "float"),
        (np.int32(5), "numpy_int", "float"),
    ],
)
def test_float_validation_type_error(fl, name, err):
    with pytest.raises(TypeError) as exc_info:
        float_validation(fl, name)
    assert str(exc_info.value).__contains__(err)


# Tests for non-finite float values
@pytest.mark.parametrize(
    "fl, name, err",
    [
        (np.inf, "infinity", "finite"),
        (-np.inf, "negative_infinity", "finite"),
        (np.nan, "nan", "finite"),
        (float("inf"), "float_infinity", "finite"),
        (float("-inf"), "float_negative_infinity", "finite"),
        (float("nan"), "float_nan", "finite"),
    ],
)
def test_float_validation_value_error(fl, name, err):
    with pytest.raises(ValueError) as exc_info:
        float_validation(fl, name)
    assert str(exc_info.value).__contains__(err)


# Tests for edge cases
@pytest.mark.parametrize(
    "fl, name",
    [
        (np.finfo(np.float64).eps, "float64_epsilon"),
        # Smallest positive float64
        (np.finfo(np.float64).max, "float64_max"),
        # Largest positive float64
        (np.finfo(np.float64).min, "float64_min"),
        # Smallest negative float64
        (1.7976931348623157e308, "max_float"),
        # Python float max
        (2.2250738585072014e-308, "min_float"),
        # Python float min
    ],
)
def test_float_validation_edge_cases(fl, name):
    # Should not raise any exception
    float_validation(fl, name)


########################################################################
# Tests for numpy_array_validation
########################################################################
# Tests for valid numpy arrays of different types and dimensions
@pytest.mark.parametrize(
    "array, name",
    [
        (np.array([1, 2, 3]), "integer_array"),
        (np.array([1.0, 2.0, 3.0]), "float_array"),
        (np.array([True, False]), "boolean_array"),
        (np.array(["a", "b", "c"]), "string_array"),
        (np.array([1 + 2j, 3 + 4j]), "complex_array"),
        (np.array([]), "empty_array"),
        (np.zeros(5), "zeros_array"),
        (np.ones(5), "ones_array"),
        (np.eye(3), "identity_array"),
        (np.array([[1, 2], [3, 4]]), "2d_array"),
        (np.array([[[1, 2], [3, 4]]]), "3d_array"),
        (np.array(5), "scalar_array"),
        (np.array(np.nan), "nan_array"),
        (np.array(np.inf), "inf_array"),
        (np.arange(10), "arange_array"),
        (np.linspace(0, 1, 5), "linspace_array"),
    ],
)
def test_numpy_array_validation_valid(array, name):
    # Should not raise any exception
    numpy_array_validation(array, name)


# Tests for invalid inputs (not numpy arrays)
@pytest.mark.parametrize(
    "array, name, err",
    [
        ([1, 2, 3], "list", "array"),
        ((1, 2, 3), "tuple", "array"),
        ({1, 2, 3}, "set", "array"),
        ({"a": 1, "b": 2}, "dict", "array"),
        (1, "integer", "array"),
        (1.0, "float", "array"),
        ("string", "string", "array"),
        (True, "boolean", "array"),
        (None, "none", "array"),
        (lambda x: x, "function", "array"),
        (np.int64(5), "numpy_scalar", "array"),
        (np.float64(5.0), "numpy_float", "array"),
    ],
)
def test_numpy_array_validation_invalid(array, name, err):
    with pytest.raises(TypeError) as exc_info:
        numpy_array_validation(array, name)
    assert str(exc_info.value).__contains__(err)


# Tests for arrays created with different numpy functions
@pytest.mark.parametrize(
    "array, name",
    [
        (np.zeros((2, 3)), "zeros_2d"),
        (np.ones((2, 2, 2)), "ones_3d"),
        (np.full((3, 3), 5), "full_array"),
        (np.identity(4), "identity_matrix"),
        (np.diag([1, 2, 3]), "diagonal_matrix"),
        (np.random.rand(3, 3), "random_array"),
        (
            np.array(np.ma.masked_array([1, 2, 3], mask=[0, 1, 0])),
            "masked_array",
        ),
        (np.array(np.matrix([[1, 2], [3, 4]])), "matrix_array"),
    ],
)
def test_numpy_array_validation_numpy_functions(array, name):
    # Should not raise any exception
    numpy_array_validation(array, name)


# Tests for arrays with different dtypes
@pytest.mark.parametrize(
    "array, name",
    [
        (np.array([1, 2, 3], dtype=np.int8), "int8_array"),
        (np.array([1, 2, 3], dtype=np.int16), "int16_array"),
        (np.array([1, 2, 3], dtype=np.int32), "int32_array"),
        (np.array([1, 2, 3], dtype=np.int64), "int64_array"),
        (np.array([1.0, 2.0, 3.0], dtype=np.float16), "float16_array"),
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), "float32_array"),
        (np.array([1.0, 2.0, 3.0], dtype=np.float64), "float64_array"),
        (np.array([True, False], dtype=np.bool_), "bool_array"),
        (np.array([1 + 2j, 3 + 4j], dtype=np.complex64), "complex64_array"),
        (np.array([1 + 2j, 3 + 4j], dtype=np.complex128), "complex128_array"),
        (np.array(["a", "b", "c"], dtype=np.unicode_), "unicode_array"),
        (np.array([b"a", b"b", b"c"], dtype=np.bytes_), "bytes_array"),
    ],
)
def test_numpy_array_validation_dtypes(array, name):
    # Should not raise any exception
    numpy_array_validation(array, name)


########################################################################
# Tests for finite_values_in_array
########################################################################
# Tests for arrays with all finite values
@pytest.mark.parametrize(
    "array, name",
    [
        (np.array([1.0, 2.0, 3.0]), "float_array"),
        (np.array([-1.0, -2.0, -3.0]), "negative_float_array"),
        (np.array([0.0]), "zero_array"),
        (np.array([1.0e-10, 2.0e-10]), "small_values_array"),
        (np.array([1.0e10, 2.0e10]), "large_values_array"),
        (np.array([]), "empty_array"),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), "2d_array"),
        (np.array([[[1.0]]]), "3d_array"),
        (np.array([np.finfo(np.float64).max]), "max_float_array"),
        (np.array([np.finfo(np.float64).min]), "min_float_array"),
        (np.array([np.finfo(np.float64).eps]), "eps_float_array"),
        (np.array([1, 2, 3], dtype=np.int32), "int_array"),
        (np.array([True, False], dtype=bool), "bool_array"),
    ],
)
def test_finite_values_in_array_valid(array, name):
    # Should not raise any exception
    finite_values_in_array(array, name)


# Tests for arrays with infinite values
@pytest.mark.parametrize(
    "array, name, err",
    [
        (np.array([1.0, np.inf, 3.0]), "inf_array", "finite"),
        (np.array([1.0, -np.inf, 3.0]), "neg_inf_array", "finite"),
        (np.array([np.inf]), "single_inf_array", "finite"),
        (np.array([-np.inf]), "single_neg_inf_array", "finite"),
        (np.array([[1.0, 2.0], [3.0, np.inf]]), "2d_inf_array", "finite"),
        (np.array([[[np.inf]]]), "3d_inf_array", "finite"),
        (np.array([np.inf, -np.inf]), "mixed_inf_array", "finite"),
        (np.array([float("inf"), 2.0, 3.0]), "py_inf_array", "finite"),
        (np.array([float("-inf"), 2.0, 3.0]), "py_neg_inf_array", "finite"),
    ],
)
def test_finite_values_in_array_infinite(array, name, err):
    with pytest.raises(ValueError) as exc_info:
        finite_values_in_array(array, name)
    assert str(exc_info.value).__contains__(err)


# Tests for arrays with NaN values
@pytest.mark.parametrize(
    "array, name, err",
    [
        (np.array([1.0, np.nan, 3.0]), "nan_array", "finite"),
        (np.array([np.nan]), "single_nan_array", "finite"),
        (np.array([[1.0, 2.0], [3.0, np.nan]]), "2d_nan_array", "finite"),
        (np.array([[[np.nan]]]), "3d_nan_array", "finite"),
        (np.array([np.nan, np.nan]), "all_nan_array", "finite"),
        (np.array([float("nan"), 2.0, 3.0]), "py_nan_array", "finite"),
    ],
)
def test_finite_values_in_array_nan(array, name, err):
    with pytest.raises(ValueError) as exc_info:
        finite_values_in_array(array, name)
    assert str(exc_info.value).__contains__(err)


# Tests for arrays with mixed non-finite values
@pytest.mark.parametrize(
    "array, name, err",
    [
        (np.array([np.inf, np.nan]), "inf_nan_array", "finite"),
        (np.array([1.0, np.inf, np.nan]), "mixed_array", "finite"),
        (np.array([[np.inf, 2.0], [3.0, np.nan]]), "2d_mixed_array", "finite"),
        (np.array([np.inf, np.nan, -np.inf]), "all_special_array", "finite"),
    ],
)
def test_finite_values_in_array_mixed(array, name, err):
    with pytest.raises(ValueError) as exc_info:
        finite_values_in_array(array, name)
    assert str(exc_info.value).__contains__(err)


########################################################################
# Tests for float_array_validation
########################################################################
# Tests for valid float arrays
@pytest.mark.parametrize(
    "array, name",
    [
        (np.array([1.0, 2.0, 3.0]), "float64_array"),
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), "float32_array"),
        (np.array([1.0, 2.0, 3.0], dtype=np.float16), "float16_array"),
        (np.array([-1.0, -2.0, -3.0]), "negative_float_array"),
        (np.array([0.0]), "zero_array"),
        (np.array([]), "empty_float_array"),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), "2d_float_array"),
        (np.array([[[1.0]]]), "3d_float_array"),
        (np.array(1.0), "scalar_float_array"),
        (np.zeros((2, 2), dtype=np.float64), "zeros_float_array"),
        (np.ones((2, 2), dtype=np.float32), "ones_float_array"),
        (np.array([np.inf, -np.inf, np.nan]), "special_values_array"),
        (
            np.array([np.finfo(np.float32).max], dtype=np.float32),
            "max_float32_array",
        ),
        (
            np.array([np.finfo(np.float64).min], dtype=np.float64),
            "min_float64_array",
        ),
    ],
)
def test_float_array_validation_valid(array, name):
    # Should not raise any exception
    float_array_validation(array, name)


# Tests for non-float arrays
@pytest.mark.parametrize(
    "array, name, err",
    [
        (np.array([1, 2, 3], dtype=np.int32), "int32_array", "float"),
        (np.array([1, 2, 3], dtype=np.int64), "int64_array", "float"),
        (np.array([True, False]), "bool_array", "float"),
        (np.array(["1.0", "2.0", "3.0"]), "string_array", "float"),
        (np.array([1 + 2j, 3 + 4j]), "complex_array", "float"),
        (np.array([1, 2, 3], dtype=np.uint8), "uint8_array", "float"),
        (np.array([[1, 2], [3, 4]], dtype=np.int16), "2d_int_array", "float"),
        (np.zeros(5, dtype=np.int32), "zeros_int_array", "float"),
        (np.ones(5, dtype=bool), "ones_bool_array", "float"),
        (np.array(5, dtype=np.int64), "scalar_int_array", "float"),
    ],
)
def test_float_array_validation_invalid_dtype(array, name, err):
    with pytest.raises(TypeError) as exc_info:
        float_array_validation(array, name)
    assert str(exc_info.value).__contains__(err)


# Tests for arrays with mixed float subclasses
@pytest.mark.parametrize(
    "array, name",
    [
        (
            np.array([np.float16(1.0), np.float32(2.0), np.float64(3.0)]),
            "mixed_float_types",
        ),
        (np.array([1.0, np.float32(2.0)]), "python_and_numpy_float"),
    ],
)
def test_float_array_validation_mixed_float_types(array, name):
    # Should not raise any exception as all are float subtypes
    float_array_validation(array, name)


########################################################################
# Tests for full_float_array_validation
########################################################################


########################################################################
# Tests for radius_validation
########################################################################
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


########################################################################
# Test for pi_validation
########################################################################
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


########################################################################
# Tests for bool_validation
########################################################################
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


########################################################################
# Tests for radii_validation
########################################################################
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
        (np.array([1, 2, 3]), "array6", ValueError, "2"),
        (np.array(5), "array7", ValueError, "2"),
        (np.array([[[1, 2], [3, 4]]]), "array9", ValueError, "2"),
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


# Testing same_size_check
@pytest.mark.parametrize(
    "arr1, name1, arr2, name2",
    [
        (np.array([]), "array1", np.array([]), "array2"),
        (np.array([1, 2, 3]), "array1", np.array([4, 5, 6]), "array2"),
        (
            np.array([[1, 2], [3, 4]]),
            "array1",
            np.array([[5, 6], [7, 8]]),
            "array2",
        ),
        (np.zeros((2, 3, 4)), "array1", np.ones((2, 3, 4)), "array2"),
        (
            np.array([1, 2, 3], dtype=int),
            "array1",
            np.array([4.0, 5.0, 6.0], dtype=float),
            "array2",
        ),
        (
            sparse.linalg.LinearOperator(
                shape=(2, 2), matvec=lambda x: 2 * np.array(x)
            ),
            "array1",
            np.array([[5, 6], [7, 8]]),
            "array2",
        ),
    ],
)
def test_same_size_check_pass(arr1, name1, arr2, name2):
    """
    Test that same_size_check does not raise an exception
    when both arrays have the same shape.
    """
    same_shape_check(arr1, name1, arr2, name2)


# Test for is_scipy_linear_op
@pytest.mark.parametrize(
    "arr1, name1, arr2, name2, expected_msg",
    [
        # Both arrays are empty but with different shapes
        (
            np.array([]),
            "array1",
            np.array([1]),
            "array2",
            "different",
        ),
        # 1D arrays with different shapes
        (
            np.array([1, 2, 3]),
            "array1",
            np.array([4, 5]),
            "array2",
            "different",
        ),
        # 2D arrays with different shapes
        (
            np.array([[1, 2], [3, 4]]),
            "array1",
            np.array([[5, 6, 7], [8, 9, 10]]),
            "array2",
            "different",
        ),
        # Higher-dimensional arrays with different shapes
        (
            np.zeros((2, 3, 4)),
            "array1",
            np.ones((2, 4, 3)),
            "array2",
            "different",
        ),
        # One empty and one non-empty array
        (
            np.array([]),
            "array1",
            np.array([[1, 2], [3, 4]]),
            "array2",
            "different",
        ),
    ],
)
def test_same_size_check_fail(arr1, name1, arr2, name2, expected_msg):
    """
    Test that same_size_check raises a ValueError
    when arrays have different shapes.
    """
    with pytest.raises(ValueError) as exc_info:
        same_shape_check(arr1, name1, arr2, name2)
    assert str(exc_info.value).__contains__(expected_msg)
    pass


# Test is_scipy_linear_op
@pytest.mark.parametrize(
    "linear_op,name",
    [
        (
            sparse.linalg.LinearOperator(
                shape=(2, 2), matvec=lambda x: 2 * np.array(x)
            ),
            "valid_op_1",
        ),
        (
            sparse.linalg.LinearOperator(shape=(3, 3), matvec=lambda x: x + 1),
            "valid_op_2",
        ),
    ],
)
def test_is_scipy_linear_op_valid_cases(linear_op, name):
    """
    Test that is_scipy_linear_op does not raise an exception for valid
    LinearOperator instances.
    """
    is_scipy_linear_op(linear_op, name)
    pass


@pytest.mark.parametrize(
    "invalid_obj,name",
    [
        # Numpy array
        (np.array([[1, 2], [3, 4]]), "array_op"),
        # Python list
        ([1, 2, 3], "list_op"),
        # Integer
        (42, "int_op"),
        # String
        ("not an operator", "string_op"),
        # None
        (None, "none_op"),
        # Dictionary
        ({"key": "value"}, "dict_op"),
    ],
)
def test_is_scipy_linear_op_invalid_cases(invalid_obj, name):
    """
    Test that is_scipy_linear_op raises a TypeError for invalid objects.
    """
    with pytest.raises(TypeError) as exc_info:
        is_scipy_linear_op(invalid_obj, name)
    pass


# Test same_type_check
@pytest.mark.parametrize(
    "array1, name1, array2, name2",
    [
        (
            np.array([1, 2, 3], dtype=np.int32),
            "array1",
            np.array([4, 5, 6], dtype=np.int32),
            "array2",
        ),
        (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "array1",
            np.array([4.0, 5.0, 6.0], dtype=np.float64),
            "array2",
        ),
        (
            np.array(["a", "b", "c"], dtype="<U1"),
            "array1",
            np.array(["x", "y", "z"], dtype="<U1"),
            "array2",
        ),
        (
            np.array([True, False, True], dtype=bool),
            "array1",
            np.array([False, False, True], dtype=bool),
            "array2",
        ),
        (
            np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
            "array1",
            np.array([5 + 6j, 7 + 8j], dtype=np.complex64),
            "array2",
        ),
        (
            np.array([1 + 2j, 3 + 4j], dtype=np.complex128),
            "array1",
            np.array([5 + 6j, 7 + 8j], dtype=np.complex128),
            "array2",
        ),
    ],
)
def test_same_type_check_pass(array1, name1, array2, name2):
    """
    Test cases where arrays have the same dtype.
    Expect no exception to be raised.
    """
    same_type_check(array1, name1, array2, name2)
    pass


@pytest.mark.parametrize(
    "array1, name1, array2, name2, expected_message",
    [
        (
            np.array([1, 2, 3], dtype=np.int32),
            "array1",
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "array2",
            "type",
        ),
        (
            np.array([1.0, 2.0], dtype=np.float32),
            "first_array",
            np.array([3.0, 4.0], dtype=np.float64),
            "second_array",
            "type",
        ),
        (
            np.array(["a", "b"], dtype="<U1"),
            "arr1",
            np.array(["c", "d"], dtype="<U2"),
            "arr2",
            "type",
        ),
        (
            np.array([True, False], dtype=bool),
            "bool_array",
            np.array([1, 0], dtype=int),
            "int_array",
            "type",
        ),
        (
            np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
            "complex_array1",
            np.array([5 + 6j, 7 + 8j], dtype=np.complex128),
            "complex_array2",
            "type",
        ),
        (
            np.array([1 + 2j, 3 + 4j], dtype=np.complex128),
            "complex_array",
            np.array([5.0, 6.0], dtype=np.float64),
            "float_array",
            "type",
        ),
        (
            np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
            "complex_array",
            np.array([1, 2, 3], dtype=np.int32),
            "int_array",
            "type",
        ),
    ],
)
def test_same_type_check_fail(array1, name1, array2, name2, expected_message):
    """
    Test cases where arrays have different dtypes.
    Expect a ValueError to be raised with the correct message.
    """
    with pytest.raises(ValueError) as exc_info:
        same_type_check(array1, name1, array2, name2)
    assert str(exc_info.value).__contains__(expected_message)
    pass


# Test for is_scipy_sparse_array
@pytest.mark.parametrize(
    "array, name",
    [
        (sparse.csr_array([[1, 0], [0, 1]]), "test_array"),
        (sparse.csc_array([[0, 2], [3, 0]]), "sparse_array"),
        (sparse.lil_array([[4, 0], [0, 5]]), "another_array"),
        (sparse.dok_array([[6, 0], [0, 7]]), "dok_array"),
        (sparse.bsr_array([[8, 0], [0, 9]]), "bsr_array"),
    ],
)
def test_is_scipy_sparse_matrix_valid(array, name):
    """
    Test that is_scipy_sparse_array does not raise an exception
    when a valid scipy sparse array is provided.
    """
    try:
        is_scipy_sparse_array(array, name)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


@pytest.mark.parametrize(
    "array, name, expected_msg",
    [
        (np.array([[1, 0], [0, 1]]), "numpy_array", "ndarray"),
        (sparse.csr_matrix([[1, 0], [0, 1]]), "test_matrix", "matrix"),
    ],
)
def test_is_scipy_sparse_matrix_invalid(array, name, expected_msg):
    """
    Test that is_scipy_sparse_matrix raises a TypeError
    with the correct message when an invalid object is provided.
    """
    with pytest.raises(TypeError) as exc_info:
        is_scipy_sparse_array(array, name)
    assert str(exc_info.value).__contains__(expected_msg)
    pass
