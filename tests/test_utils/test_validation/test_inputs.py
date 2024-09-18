import pytest
import numpy as np
import warnings
from biosspheres.utils.validation.inputs import (
    big_l_validation,
    radius_validation,
    bool_validation,
    radii_validation,
)


# Tests for big_l_validation
@pytest.mark.parametrize("big_l, name", [
    (0, "big_l"),
    (1500, "big_l"),
    (3000, "big_l"),
    (np.int32(100), "big_l"),
])
def test_big_l_validation_valid(big_l, name):
    # Should not raise any exception
    big_l_validation(big_l, name)


@pytest.mark.parametrize("big_l, name", [
    ("100", "big_l"),
    (10.5, "big_l"),
    (None, "big_l"),
    ([100], "big_l"),
])
def test_big_l_validation_invalid_type(big_l, name):
    with pytest.raises(TypeError) as exc_info:
        big_l_validation(big_l, name)
    assert f"{name} must be an integer or numpy integer type" in str(exc_info.value)


@pytest.mark.parametrize("big_l, name", [
    (-1, "big_l"),
    (-100, "big_l"),
])
def test_big_l_validation_negative(big_l, name):
    with pytest.raises(ValueError) as exc_info:
        big_l_validation(big_l, name)
    assert f"{name} must be non-negative" in str(exc_info.value)


def test_big_l_validation_warning():
    big_l = 3001
    name = "big_l"
    with pytest.warns(UserWarning) as record:
        big_l_validation(big_l, name)
    assert len(record) == 1
    assert f"{name} is {big_l}, there might be issues with {name} being too big." in str(record[0].message)


# Tests for radius_validation
@pytest.mark.parametrize("r, name", [
    (1.0, "radius"),
    (np.float64(2.5), "radius"),
])
def test_radius_validation_valid(r, name):
    # Should not raise any exception
    radius_validation(r, name)


@pytest.mark.parametrize("r, name", [
    (2, "pi"),
    ("1.0", "radius"),
    (True, "radius"),
    (None, "radius"),
    ([1.0], "radius"),
])
def test_radius_validation_invalid_type(r, name):
    with pytest.raises(TypeError) as exc_info:
        radius_validation(r, name)
    assert f"{name} must be a float or numpy float type" in str(exc_info.value)


@pytest.mark.parametrize("r, name, error_msg", [
    (np.inf, "radius", "must be a finite number"),
    (-np.inf, "radius", "must be a finite number"),
    (np.nan, "radius", "must be a finite number"),
    (0.0, "radius", "must be positive"),
    (-1.0, "radius", "must be positive"),
])
def test_radius_validation_invalid_value(r, name, error_msg):
    with pytest.raises(ValueError) as exc_info:
        radius_validation(r, name)
    assert error_msg in str(exc_info.value)


# Test for pi_validation
@pytest.mark.parametrize("pi, name", [
    (2, "pi"),
    ("1.0", "pi"),
    (True, "pi"),
    (None, "pi"),
    ([1.0], "pi"),
])
def test_radius_validation_invalid_type(pi, name):
    with pytest.raises(TypeError) as exc_info:
        radius_validation(pi, name)
    assert f"{name} must be a float or numpy float type" in str(exc_info.value)


@pytest.mark.parametrize("pi, name, error_msg", [
    (np.inf, "pi", "must be a finite number"),
    (-np.inf, "pi", "must be a finite number"),
    (np.nan, "pi", "must be a finite number"),
])
def test_radius_validation_invalid_value(pi, name, error_msg):
    with pytest.raises(ValueError) as exc_info:
        radius_validation(pi, name)
    assert error_msg in str(exc_info.value)


# Tests for bool_validation
@pytest.mark.parametrize("b, name", [
    (True, "flag"),
    (False, "flag"),
    (np.bool_(True), "flag"),
    (np.bool_(False), "flag"),
])
def test_bool_validation_valid(b, name):
    # Should not raise any exception
    bool_validation(b, name)


@pytest.mark.parametrize("b, name", [
    ("True", "flag"),
    (1, "flag"),
    (0, "flag"),
    (None, "flag"),
    ([True], "flag"),
])
def test_bool_validation_invalid_type(b, name):
    with pytest.raises(TypeError) as exc_info:
        bool_validation(b, name)
    assert f"{name} must be a boolean or numpy boolean type" in str(exc_info.value)


# Tests for radii_validation
@pytest.mark.parametrize("radii, name", [
    (np.array([1.0, 2.5, 3.3], dtype=np.float64), "radii"),
    ([1.0, 2.5, 3.3], "radii"),  # Will be converted to np.ndarray
])
def test_radii_validation_valid(radii, name):
    if not isinstance(radii, np.ndarray):
        with pytest.warns(UserWarning) as record:
            radii_validation(radii, name)
        assert len(record) == 1
        assert f"{name} was not a numpy array." in str(record[0].message)
    else:
        # Should not raise any exception
        radii_validation(radii, name)


@pytest.mark.parametrize("radii, name", [
    ("not an array", "radii"),
    (123, "radii"),
    (None, "radii"),
    ({1: "a"}, "radii"),
])
def test_radii_validation_invalid_type(radii, name):
    with pytest.raises(TypeError) as exc_info:
        radii_validation(radii, name)


@pytest.mark.parametrize("radii, name", [
    (np.array([1, 2, 3], dtype=np.int32), "radii"),  # Not float dtype
    (np.array([1.0, 2.5, "3.3"], dtype=object), "radii"),
])
def test_radii_validation_invalid_dtype(radii, name):
    with pytest.raises(TypeError) as exc_info:
        radii_validation(radii, name)
    assert f"{name} must be an array of floats" in str(exc_info.value)


def test_radii_validation_empty_array():
    radii = np.array([], dtype=np.float64)
    name = "radii"
    with pytest.raises(ValueError) as exc_info:
        radii_validation(radii, name)
    assert f"{name} array must not be empty" in str(exc_info.value)


@pytest.mark.parametrize("radii, name", [
    (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), "radii"),
    (np.array([1.0, 2.0, 3.0], dtype=np.float64).reshape(-1, 1), "radii"),
])
def test_radii_validation_wrong_dimensions(radii, name):
    with pytest.raises(ValueError) as exc_info:
        radii_validation(radii, name)
    assert f"Expected '{name}' to be one-dimensional" in str(exc_info.value)


@pytest.mark.parametrize("radii, name", [
    (np.array([1.0, np.inf, 3.0], dtype=np.float64), "radii"),
    (np.array([1.0, np.nan, 3.0], dtype=np.float64), "radii"),
])
def test_radii_validation_non_finite_elements(radii, name):
    with pytest.raises(ValueError) as exc_info:
        radii_validation(radii, name)
    assert f"All elements in {name} must be a finite number" in str(exc_info.value)


@pytest.mark.parametrize("radii, name", [
    (np.array([1.0, -2.5, 3.3], dtype=np.float64), "radii"),
    (np.array([0.0, 2.0, 3.0], dtype=np.float64), "radii"),
])
def test_radii_validation_non_positive_elements(radii, name):
    with pytest.raises(ValueError) as exc_info:
        radii_validation(radii, name)
    assert f"All elements in {name} must be positive numbers" in str(exc_info.value)
