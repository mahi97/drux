# -*- coding: utf-8 -*-
"""Shared numerical core for time grids and array-native model evaluation."""

from typing import Any, Callable, Tuple, Union

import numpy as np

from .messages import (
    ERROR_DURATION_TIME_STEP_FINITE,
    ERROR_DURATION_TIME_STEP_POSITIVE,
    ERROR_PARAMETERS_NONFINITE,
    ERROR_TIME_NEGATIVE,
    ERROR_TIME_NONFINITE,
    ERROR_TIME_NUMERIC,
    ERROR_TIME_STEP_GREATER_THAN_DURATION,
)

# Relative tolerance used to decide whether duration is an integer number of steps.
_TIME_GRID_ATOL_FACTOR = 1e-9

Number = Union[int, float, np.number]
TimeValue = Union[float, np.ndarray]


def _require_real_scalar(value: Any, error_message: str) -> float:
    """
    Coerce a value to a real Python float.

    :param value: candidate scalar
    :param error_message: raised when the value is not a real number
    """
    if isinstance(value, bool) or isinstance(value, (str, bytes)) or value is None:
        raise ValueError(error_message)
    if not np.isscalar(value):
        raise ValueError(error_message)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(error_message)
    return number


def validate_simulation_schedule(duration: Any, time_step: Any) -> Tuple[float, float]:
    """
    Validate duration and time-step arguments for simulate().

    :param duration: total simulation time (s)
    :param time_step: spacing of interior sample points (s)
    :return: duration and time step as floats
    :raises ValueError: if either value is non-numeric, non-finite, non-positive,
        or if time_step is greater than duration
    """
    duration_value = _require_real_scalar(duration, ERROR_DURATION_TIME_STEP_POSITIVE)
    time_step_value = _require_real_scalar(time_step, ERROR_DURATION_TIME_STEP_POSITIVE)
    if not (np.isfinite(duration_value) and np.isfinite(time_step_value)):
        raise ValueError(ERROR_DURATION_TIME_STEP_FINITE)
    if duration_value <= 0 or time_step_value <= 0:
        raise ValueError(ERROR_DURATION_TIME_STEP_POSITIVE)
    if time_step_value > duration_value:
        raise ValueError(ERROR_TIME_STEP_GREATER_THAN_DURATION)
    return duration_value, time_step_value


def build_time_grid(duration: Number, time_step: Number) -> np.ndarray:
    """
    Build a simulation timeline that starts at 0 and ends at duration.

    Interior points are spaced by ``time_step``. When ``duration`` is not an
    integer multiple of ``time_step``, the last interior point that still lies
    strictly before ``duration`` is kept and ``duration`` itself is appended.
    The returned array never contains a value after ``duration``.

    :param duration: total simulation time (s); must already be validated
    :param time_step: spacing of interior sample points (s)
    :return: 1-D float array of time points
    """
    duration_value = float(duration)
    time_step_value = float(time_step)
    n_exact = duration_value / time_step_value
    n_steps = int(round(n_exact))
    atol = _TIME_GRID_ATOL_FACTOR * max(duration_value, time_step_value)
    if n_steps >= 1 and abs(n_steps * time_step_value - duration_value) <= atol:
        return np.linspace(0.0, duration_value, n_steps + 1)

    n_fit = int(np.floor(n_exact))
    if n_fit < 0:
        n_fit = 0
    interior = np.arange(n_fit + 1, dtype=float) * time_step_value
    interior = interior[interior < duration_value]
    if interior.size == 0:
        return np.array([0.0, duration_value], dtype=float)
    return np.concatenate([interior, np.array([duration_value], dtype=float)])


def as_time_array(t: Any) -> Tuple[np.ndarray, bool]:
    """
    Convert a scalar or array-like time input into a float ndarray.

    :param t: time in seconds, scalar or array-like
    :return: ``(array, is_scalar)`` where ``is_scalar`` is True when ``t`` was
        a 0-D value that should be returned as a Python float
    :raises ValueError: if any value is non-numeric, non-finite, or negative
    """
    if isinstance(t, bool) or isinstance(t, (str, bytes)) or t is None:
        raise ValueError(ERROR_TIME_NUMERIC)
    try:
        raw = np.asarray(t)
        if np.iscomplexobj(raw):
            raise ValueError(ERROR_TIME_NUMERIC)
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(ERROR_TIME_NUMERIC)

    is_scalar = np.isscalar(t) or array.ndim == 0
    if is_scalar:
        array = np.reshape(array, (1,))

    if array.size and not np.all(np.isfinite(array)):
        raise ValueError(ERROR_TIME_NONFINITE)
    if array.size and np.any(array < 0):
        raise ValueError(ERROR_TIME_NEGATIVE)
    return array, bool(is_scalar)


def evaluate_model(
        model_function: Callable[[np.ndarray], Any],
        t: Any) -> TimeValue:
    """
    Evaluate a vectorized model function at a scalar or array of times.

    :param model_function: callable that accepts a float ndarray and returns
        an array-like result of the same shape
    :param t: time in seconds, scalar or array-like
    :return: Python float for a scalar input, otherwise an ndarray
    """
    time_array, is_scalar = as_time_array(t)
    result = np.asarray(model_function(time_array), dtype=float)
    if is_scalar:
        return float(np.reshape(result, ()))
    return result


def validate_finite_parameters(parameters: Any) -> None:
    """
    Reject non-finite numeric fields on a parameter object.

    Scientific range checks stay in each model; this only enforces that stored
    numeric values are finite.

    :param parameters: dataclass (or similar) of model parameters
    :raises ValueError: if a numeric field is NaN or infinite
    """
    if parameters is None:
        return
    try:
        fields = vars(parameters)
    except TypeError:
        return
    for value in fields.values():
        if isinstance(value, bool) or isinstance(value, (str, bytes)):
            continue
        if isinstance(value, (int, float, np.number)):
            if not np.isfinite(float(value)):
                raise ValueError(ERROR_PARAMETERS_NONFINITE)
