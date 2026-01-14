"""
Core J-PARSE algorithm implementation.

This module contains the pure J-PARSE algorithm that only requires numpy.
It operates directly on Jacobian matrices without any robot kinematics dependencies.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray


class JParseCore:
    """
    Pure J-PARSE algorithm implementation.

    J-PARSE (Jacobian-based Projection Algorithm for Resolving Singularities Effectively)
    provides singularity-aware inverse kinematics by computing a modified pseudo-inverse
    that handles singular configurations smoothly.

    This class operates directly on Jacobian matrices and requires only numpy.
    For URDF-based robot kinematics, use the Robot class instead.

    Parameters
    ----------
    gamma : float, optional
        Singularity threshold (0 < gamma < 1). Directions with adjusted condition
        numbers below this threshold are treated as singular. Default is 0.1.

    Examples
    --------
    >>> import numpy as np
    >>> from jparse_robotics import JParseCore
    >>>
    >>> # Create J-PARSE solver
    >>> jparse = JParseCore(gamma=0.1)
    >>>
    >>> # 2-link planar arm Jacobian
    >>> J = np.array([[-0.707, -0.707], [0.707, 0.707]])
    >>>
    >>> # Compute J-PARSE pseudo-inverse
    >>> J_parse = jparse.compute(J)
    >>>
    >>> # Use for velocity control
    >>> desired_velocity = np.array([0.1, 0.1])
    >>> joint_velocities = J_parse @ desired_velocity
    """

    def __init__(self, gamma: float = 0.1) -> None:
        if not 0 < gamma < 1:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        self.gamma = gamma

    def _svd_compose(
        self, U: NDArray[np.floating], S: list, Vt: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Recompose a matrix from its SVD components.

        Parameters
        ----------
        U : ndarray
            Left singular vectors (m x k matrix)
        S : list
            Singular values
        Vt : ndarray
            Right singular vectors transposed (n x n matrix)

        Returns
        -------
        ndarray
            Recomposed matrix U @ diag(S) @ Vt
        """
        Sfull = np.zeros((U.shape[1], Vt.shape[0]))
        for i in range(min(len(S), Sfull.shape[0], Sfull.shape[1])):
            Sfull[i, i] = S[i]
        return np.asarray(np.matrix(U) @ Sfull @ np.matrix(Vt))

    def compute(
        self,
        jacobian: NDArray[np.floating],
        singular_direction_gain_position: float = 1.0,
        singular_direction_gain_angular: float = 1.0,
        position_dimensions: int | None = None,
        angular_dimensions: int | None = None,
        return_nullspace: bool = False,
    ) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], NDArray[np.floating]]]:
        """
        Compute the J-PARSE pseudo-inverse of a Jacobian matrix.

        The J-PARSE algorithm decomposes the Jacobian using SVD and constructs
        a modified pseudo-inverse that:
        1. Clamps singular values below gamma*sigma_max (safety Jacobian)
        2. Projects commands onto non-singular directions (projection Jacobian)
        3. Provides smooth feedback in singular directions (singular direction term)

        Parameters
        ----------
        jacobian : ndarray
            The m x n Jacobian matrix to invert.
        singular_direction_gain_position : float, optional
            Gain for position-related singular directions. Default is 1.0.
        singular_direction_gain_angular : float, optional
            Gain for orientation-related singular directions. Default is 1.0.
        position_dimensions : int, optional
            Number of rows corresponding to position (typically 3 for 3D).
            If None and angular_dimensions is None, all rows use position gain.
        angular_dimensions : int, optional
            Number of rows corresponding to orientation (typically 3 for 3D).
            If None and position_dimensions is set, only position gain is used.
        return_nullspace : bool, optional
            If True, also return the nullspace projection matrix. Default is False.

        Returns
        -------
        J_parse : ndarray
            The n x m J-PARSE pseudo-inverse matrix.
        nullspace : ndarray, optional
            The n x n nullspace projection matrix (only if return_nullspace=True).
            Can be used for secondary objectives that don't interfere with the
            primary task.

        Notes
        -----
        The J-PARSE formula is:
            J_parse = J_safety^+ @ J_proj @ J_proj^+ + J_safety^+ @ Phi_singular

        Where:
        - J_safety: Jacobian with singular values clamped to gamma*sigma_max
        - J_proj: Jacobian with only non-singular directions
        - Phi_singular: Smooth feedback term for singular directions
        """
        J = np.asarray(jacobian)

        # SVD decomposition
        U, S, Vt = np.linalg.svd(J)
        sigma_max = np.max(S)

        # Adjusted condition numbers for each singular value
        adjusted_condition_numbers = [sig / sigma_max for sig in S]

        # Build projection Jacobian (only non-singular directions)
        U_new_proj = []
        S_new_proj = []
        for col in range(len(S)):
            if S[col] > self.gamma * sigma_max:
                U_new_proj.append(np.matrix(U[:, col]).T)
                S_new_proj.append(S[col])

        if len(U_new_proj) > 0:
            U_new_proj_arr = np.concatenate(U_new_proj, axis=1)
            J_proj = self._svd_compose(U_new_proj_arr, S_new_proj, Vt)
        else:
            # All directions are singular - use safety Jacobian only
            J_proj = J.copy()

        # Build safety Jacobian (clamp singular values)
        S_new_safety = [
            s if (s / sigma_max) > self.gamma else self.gamma * sigma_max for s in S
        ]
        J_safety = self._svd_compose(U, S_new_safety, Vt)

        # Build singular direction feedback term
        U_new_sing = []
        Phi = []
        has_singular_directions = False

        for col in range(len(S)):
            if adjusted_condition_numbers[col] <= self.gamma:
                has_singular_directions = True
                U_new_sing.append(np.matrix(U[:, col]).T)
                # Smooth transition: s/(s_max * gamma)
                Phi.append(adjusted_condition_numbers[col] / self.gamma)

        Phi_singular = np.zeros(U.shape)

        if has_singular_directions:
            U_new_sing_arr = np.matrix(np.concatenate(U_new_sing, axis=1))
            Phi_mat = np.matrix(np.diag(Phi))

            # Determine gain matrix
            if position_dimensions is None and angular_dimensions is None:
                # Default: use position gain for all dimensions
                gain_dimension = J.shape[0]
                gains = np.array(
                    [singular_direction_gain_position] * gain_dimension, dtype=float
                )
            elif angular_dimensions is None and position_dimensions is not None:
                # Only position dimensions specified
                gains = np.array(
                    [singular_direction_gain_position] * position_dimensions, dtype=float
                )
            elif position_dimensions is None and angular_dimensions is not None:
                # Only angular dimensions specified
                gains = np.array(
                    [singular_direction_gain_angular] * angular_dimensions, dtype=float
                )
            else:
                # Both specified
                gains = np.array(
                    [singular_direction_gain_position] * position_dimensions
                    + [singular_direction_gain_angular] * angular_dimensions,
                    dtype=float,
                )

            Kp_singular = np.diag(gains)
            Phi_singular = U_new_sing_arr @ Phi_mat @ U_new_sing_arr.T @ Kp_singular

        # Compute pseudo-inverses
        J_safety_pinv = np.linalg.pinv(J_safety)
        J_proj_pinv = np.linalg.pinv(J_proj)

        # Compute J-PARSE
        if has_singular_directions:
            J_parse = np.asarray(
                J_safety_pinv @ J_proj @ J_proj_pinv + J_safety_pinv @ Phi_singular
            )
        else:
            J_parse = np.asarray(J_safety_pinv @ J_proj @ J_proj_pinv)

        if return_nullspace:
            J_safety_nullspace = np.eye(J_safety.shape[1]) - J_safety_pinv @ J_safety
            return J_parse, np.asarray(J_safety_nullspace)

        return J_parse

    def pinv(self, jacobian: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the standard Moore-Penrose pseudo-inverse.

        This is provided for comparison with J-PARSE.

        Parameters
        ----------
        jacobian : ndarray
            The m x n Jacobian matrix.

        Returns
        -------
        ndarray
            The n x m pseudo-inverse matrix.
        """
        return np.linalg.pinv(jacobian)

    def damped_least_squares(
        self,
        jacobian: NDArray[np.floating],
        damping: float = 0.01,
        return_nullspace: bool = False,
    ) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], NDArray[np.floating]]]:
        """
        Compute the damped least squares (DLS) pseudo-inverse.

        Also known as Levenberg-Marquardt or SR-inverse. This is a common
        baseline method for comparison with J-PARSE.

        Parameters
        ----------
        jacobian : ndarray
            The m x n Jacobian matrix.
        damping : float, optional
            Damping factor (lambda). Default is 0.01.
        return_nullspace : bool, optional
            If True, also return the nullspace projection matrix.

        Returns
        -------
        J_dls : ndarray
            The n x m damped least squares pseudo-inverse.
        nullspace : ndarray, optional
            The n x n nullspace projection matrix (only if return_nullspace=True).
        """
        J = np.asarray(jacobian)
        J_dls = np.linalg.inv(J.T @ J + damping**2 * np.eye(J.shape[1])) @ J.T

        if return_nullspace:
            nullspace = np.eye(J.shape[1]) - J_dls @ J
            return J_dls, nullspace

        return J_dls


def manipulability_measure(jacobian: NDArray[np.floating]) -> float:
    """
    Compute Yoshikawa's manipulability measure.

    The manipulability measure indicates how well-conditioned the Jacobian is
    at a given configuration. Higher values indicate better manipulability.

    Parameters
    ----------
    jacobian : ndarray
        The m x n Jacobian matrix.

    Returns
    -------
    float
        The manipulability measure: sqrt(det(J @ J.T))
    """
    J = np.asarray(jacobian)
    return float(np.sqrt(np.linalg.det(J @ J.T)))


def inverse_condition_number(jacobian: NDArray[np.floating]) -> float:
    """
    Compute the inverse condition number of the Jacobian.

    The inverse condition number (sigma_min / sigma_max) indicates proximity
    to singularity. Values close to 0 indicate near-singular configurations,
    while values close to 1 indicate well-conditioned configurations.

    Parameters
    ----------
    jacobian : ndarray
        The m x n Jacobian matrix.

    Returns
    -------
    float
        The inverse condition number (sigma_min / sigma_max).
    """
    J = np.asarray(jacobian)
    _, S, _ = np.linalg.svd(J)
    return float(np.min(S) / np.max(S))
