"""This file contains the abstract base class for drug release models."""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from .messages import (
    ERROR_TARGET_RELEASE_RANGE,
    ERROR_NO_SIMULATION_DATA,
    ERROR_RELEASE_PROFILE_TOO_SHORT,
    ERROR_TARGET_RELEASE_EXCEEDS_MAX,
)
from .numerical import (
    build_time_grid,
    evaluate_model,
    validate_finite_parameters,
    validate_simulation_schedule,
)


class DrugReleaseModel(ABC):
    """
    Abstract base class for drug release models.

    This class provides a common interface and functionality for various
    mathematical models of drug release from delivery systems.

    Shared numerical behavior (time-grid construction, scalar/array
    evaluation, and transactional updates of simulation state) lives in
    the numerical core. Subclasses should implement only:

    - _model_function(): Vectorized core model equation
    - _validate_parameters(): Model-specific scientific restrictions
    """

    def __init__(self):
        """Initialize the drug release model."""
        self._time_points = None
        self._release_profile = None
        self._plot_parameters = {
            "xlabel": "Time (s)",
            "ylabel": "Cumulative Release",
            "title": "Drug Release Profile",
            "label": "Release Profile"}

    @abstractmethod
    def _validate_parameters(self) -> None:
        """
        Validate model parameters.

        Should raise ValueError if parameters are invalid.
        """
        pass

    @abstractmethod
    def _model_function(self, t: np.ndarray) -> np.ndarray:
        """
        Vectorized model equation for cumulative drug release.

        Implementations must accept a NumPy ndarray of times (seconds) and
        return an ndarray of the same shape. Element-wise NumPy arithmetic
        also accepts Python scalars.

        :param t: time point(s) at which to calculate drug release
        """
        pass

    def _check_parameters(self) -> None:
        """Run shared finite-value checks, then model-specific restrictions."""
        validate_finite_parameters(getattr(self, "_parameters", None))
        self._validate_parameters()

    def _get_release_profile(self, time_points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the drug release profile over the given time points.

        :param time_points: times to evaluate; defaults to the stored grid
        """
        t = self._time_points if time_points is None else time_points
        return np.asarray(self._model_function(t), dtype=float)

    def evaluate(self, t: Any) -> Union[float, np.ndarray]:
        """
        Evaluate the model at one or more time points.

        This method does not change the last successful simulation. A Python
        float is returned for a scalar time; an ndarray is returned for
        array-like input.

        :param t: time in seconds, scalar or array-like
        :return: cumulative release at ``t``
        :raises ValueError: if parameters or times are invalid
        """
        self._check_parameters()
        return evaluate_model(self._model_function, t)

    def _validate_plot(self) -> tuple:
        """
        Validate plotting process.

        :raises ValueError: if simulation data is not available
        :raises ValueError: if release profile is too short
        """
        if self._time_points is None or self._release_profile is None:
            raise ValueError(ERROR_NO_SIMULATION_DATA)

        if len(self._release_profile) < 2:
            raise ValueError(ERROR_RELEASE_PROFILE_TOO_SHORT)
        fig, ax = plt.subplots()
        return fig, ax

    def simulate(self, duration: float, time_step: float = 1) -> np.ndarray:
        """
        Simulate drug release over time.

        The generated timeline always starts at 0 and always finishes at
        exactly ``duration``. Interior points are spaced by ``time_step``.
        When ``duration`` is not divisible by ``time_step``, the last
        interior point that is still strictly before ``duration`` is kept
        and ``duration`` itself is appended. No sample is placed after
        ``duration``.

        Simulation state (``_time_points`` and ``_release_profile``) is
        updated only after a complete, successful calculation. A failed
        call leaves the previous successful result unchanged.

        :param duration: total time for simulation (in seconds)
        :param time_step: time step for simulation (in seconds)
        :return: cumulative release at each time point
        """
        duration_value, time_step_value = validate_simulation_schedule(
            duration, time_step)
        self._check_parameters()
        time_points = build_time_grid(duration_value, time_step_value)
        release_profile = self._get_release_profile(time_points)
        self._time_points = time_points
        self._release_profile = release_profile
        return self._release_profile

    def plot(
            self,
            show: bool = True,
            label: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            title: Optional[str] = None,
            **kwargs: Any) -> tuple:
        """
        Plot the drug release profile.

        :param show: Whether to display the plot (default: True)
        :param label: The legend label for the release profile curve
        :param xlabel: Label for the x-axis
        :param ylabel: Label for the y-axis
        :param title: Title of the plot
        """
        # Create a new figure and axis if not provided
        fig, ax = self._validate_plot()

        # Plotting the release profile
        ax.plot(
            self._time_points, self._release_profile, label=label or self._plot_parameters["label"], **kwargs
        )
        ax.set_xlabel(xlabel or self._plot_parameters["xlabel"])
        ax.set_ylabel(ylabel or self._plot_parameters["ylabel"])
        ax.set_title(title or self._plot_parameters["title"])
        ax.grid()
        ax.legend()

        # Show the plot if requested
        if show:
            fig.show()

        return fig, ax

    def get_release_rate(self) -> np.ndarray:
        """Calculate the instantaneous release rate (derivative of release profile)."""
        if self._time_points is None or self._release_profile is None:
            raise ValueError(ERROR_NO_SIMULATION_DATA)

        if len(self._release_profile) < 2:
            raise ValueError(ERROR_RELEASE_PROFILE_TOO_SHORT)

        # Calculate the derivative of the release profile
        release_rate = np.gradient(self._release_profile, self._time_points)
        return release_rate

    def time_for_release(self, target_release: float) -> float:
        """
        Estimate time needed to reach a specific release percentage.

        :param target_release: target release fraction (>= 0)

        :raises ValueError: if target_release is negative
        :raises ValueError: if simulation data is not available
        :raises ValueError: if target_release exceeds maximum release
        """
        if self._time_points is None or self._release_profile is None:
            raise ValueError(ERROR_NO_SIMULATION_DATA)

        if target_release < 0:
            raise ValueError(ERROR_TARGET_RELEASE_RANGE)

        if target_release > self._release_profile[-1]:
            raise ValueError(ERROR_TARGET_RELEASE_EXCEEDS_MAX)

        # Find first time point where release >= target
        idx = np.argmax(self._release_profile >= target_release)
        return self._time_points[idx]
