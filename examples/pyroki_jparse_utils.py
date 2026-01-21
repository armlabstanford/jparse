#!/usr/bin/env python3
"""
Utility module bridging pyroki and J-PARSE.

This module provides utilities for using J-PARSE singularity-aware inverse
kinematics with pyroki's differentiable robot kinematics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
from jax.typing import ArrayLike


def compute_jacobian_from_pyroki(
    robot: pk.Robot,
    cfg: ArrayLike,
    target_link_index: int,
    position_only: bool = True,
) -> np.ndarray:
    """
    Extract the Jacobian matrix from a pyroki robot using JAX autodiff.

    Parameters
    ----------
    robot : pk.Robot
        The pyroki robot model.
    cfg : ArrayLike
        Joint configuration (actuated_count,).
    target_link_index : int
        Index of the target link in robot.links.names.
    position_only : bool, optional
        If True, return only the 3xn position Jacobian.
        If False, return the full 6xn Jacobian (position + orientation).
        Default is True.

    Returns
    -------
    np.ndarray
        The Jacobian matrix. Shape is (3, n) if position_only else (6, n).
    """
    cfg = jnp.asarray(cfg)

    if position_only:
        # Position Jacobian via autodiff on translation
        jacobian = jax.jacfwd(
            lambda q: jaxlie.SE3(robot.forward_kinematics(q)).translation()
        )(cfg)[target_link_index]
    else:
        # Full Jacobian via autodiff on SE3 log
        def get_pose_log(q):
            poses = robot.forward_kinematics(q)
            target_pose = jaxlie.SE3(poses[target_link_index])
            return target_pose.log()

        jacobian = jax.jacfwd(get_pose_log)(cfg)

    return np.array(jacobian)


def get_link_pose(
    robot: pk.Robot,
    cfg: ArrayLike,
    target_link_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the position and quaternion of a target link.

    Parameters
    ----------
    robot : pk.Robot
        The pyroki robot model.
    cfg : ArrayLike
        Joint configuration.
    target_link_index : int
        Index of the target link.

    Returns
    -------
    position : np.ndarray
        Position (3,).
    wxyz : np.ndarray
        Quaternion in wxyz format (4,).
    """
    cfg = jnp.asarray(cfg)
    poses = robot.forward_kinematics(cfg)
    target_pose = jaxlie.SE3(poses[target_link_index])
    position = np.array(target_pose.translation().squeeze())
    wxyz = np.array(target_pose.rotation().wxyz)
    return position, wxyz


class JParsePyrokiController:
    """
    Velocity-level IK controller using J-PARSE with pyroki robots.

    This controller computes joint velocities from desired task-space velocities
    using J-PARSE's singularity-aware pseudo-inverse.

    Parameters
    ----------
    robot : pk.Robot
        The pyroki robot model.
    target_link_name : str
        Name of the end-effector link.
    gamma : float, optional
        J-PARSE singularity threshold (0 < gamma < 1). Default is 0.1.
    position_only : bool, optional
        If True, use only position control (3 DoF). Default is True.

    Examples
    --------
    >>> import pyroki as pk
    >>> from robot_descriptions.loaders.yourdfpy import load_robot_description
    >>> from pyroki_jparse_utils import JParsePyrokiController
    >>>
    >>> urdf = load_robot_description("panda_description")
    >>> robot = pk.Robot.from_urdf(urdf)
    >>> controller = JParsePyrokiController(robot, "panda_hand", gamma=0.1)
    >>>
    >>> cfg = np.zeros(7)
    >>> target_pos = np.array([0.5, 0.0, 0.4])
    >>> dq = controller.compute_velocity(cfg, target_pos)
    """

    def __init__(
        self,
        robot: pk.Robot,
        target_link_name: str,
        gamma: float = 0.1,
        position_only: bool = True,
    ):
        from jparse_robotics import JParseCore

        self.robot = robot
        self.target_link_name = target_link_name
        self.target_link_index = robot.links.names.index(target_link_name)
        self.position_only = position_only
        self.gamma = gamma

        # Create J-PARSE solver
        self.jparse = JParseCore(gamma=gamma)

        # Control gains
        self.position_gain = 5.0
        self.orientation_gain = 1.0  # Lower gain for smoother orientation control

        # Singular direction gains (for exiting singularities)
        # Higher values = more aggressive motion in singular directions
        self.singular_direction_gain_position = 1.0
        self.singular_direction_gain_angular = 1.0

        # Nullspace control gains
        self.nullspace_gain = 0.5  # Gain for nullspace motion toward home
        self.nullspace_enabled = True

        # Velocity limits
        self.max_joint_velocity = 2.0  # rad/s

        # Method selection
        self._method = "jparse"  # "jparse", "pinv", or "dls"
        self._dls_damping = 0.05

        # Home configuration (middle of joint range by default)
        self._home_cfg = None

    @property
    def method(self) -> str:
        """Current IK method: 'jparse', 'pinv', or 'dls'."""
        return self._method

    @method.setter
    def method(self, value: str):
        if value not in ["jparse", "pinv", "dls"]:
            raise ValueError(f"method must be 'jparse', 'pinv', or 'dls', got {value}")
        self._method = value

    def get_jacobian(self, cfg: ArrayLike) -> np.ndarray:
        """
        Compute the Jacobian at the current configuration.

        Parameters
        ----------
        cfg : ArrayLike
            Joint configuration.

        Returns
        -------
        np.ndarray
            The Jacobian matrix.
        """
        return compute_jacobian_from_pyroki(
            self.robot, cfg, self.target_link_index, self.position_only
        )

    def get_current_pose(self, cfg: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector pose.

        Parameters
        ----------
        cfg : ArrayLike
            Joint configuration.

        Returns
        -------
        position : np.ndarray
            End-effector position (3,).
        wxyz : np.ndarray
            End-effector orientation as quaternion (4,).
        """
        return get_link_pose(self.robot, cfg, self.target_link_index)

    def compute_inverse(
        self, jacobian: np.ndarray, return_nullspace: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute the pseudo-inverse using the selected method.

        Parameters
        ----------
        jacobian : np.ndarray
            The Jacobian matrix.
        return_nullspace : bool, optional
            If True, also return the nullspace projector. Default is False.

        Returns
        -------
        J_inv : np.ndarray
            The pseudo-inverse matrix.
        J_nullspace : np.ndarray or None
            The nullspace projector (I - J_inv @ J), or None if not requested.
        """
        J_nullspace = None

        if self._method == "jparse":
            # Pass singular direction gains for better singularity exit behavior
            position_dims = 3 if self.position_only else 3
            angular_dims = 0 if self.position_only else 3
            if return_nullspace:
                J_inv, J_nullspace = self.jparse.compute(
                    jacobian,
                    singular_direction_gain_position=self.singular_direction_gain_position,
                    singular_direction_gain_angular=self.singular_direction_gain_angular,
                    position_dimensions=position_dims,
                    angular_dimensions=angular_dims,
                    return_nullspace=True,
                )
            else:
                J_inv = self.jparse.compute(
                    jacobian,
                    singular_direction_gain_position=self.singular_direction_gain_position,
                    singular_direction_gain_angular=self.singular_direction_gain_angular,
                    position_dimensions=position_dims,
                    angular_dimensions=angular_dims,
                )
        elif self._method == "pinv":
            J_inv = self.jparse.pinv(jacobian)
            if return_nullspace:
                J_nullspace = np.eye(jacobian.shape[1]) - J_inv @ jacobian
        else:  # dls
            J_inv = self.jparse.damped_least_squares(jacobian, self._dls_damping)
            if return_nullspace:
                J_nullspace = np.eye(jacobian.shape[1]) - J_inv @ jacobian

        return J_inv, J_nullspace

    def compute_velocity(
        self,
        cfg: ArrayLike,
        target_position: np.ndarray,
        target_wxyz: Optional[np.ndarray] = None,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute joint velocities to move toward target pose.

        Parameters
        ----------
        cfg : ArrayLike
            Current joint configuration.
        target_position : np.ndarray
            Target position (3,).
        target_wxyz : np.ndarray, optional
            Target orientation as quaternion (4,). Only used if position_only=False.
        dt : float, optional
            Time step for integration. Default is 0.02.

        Returns
        -------
        dq : np.ndarray
            Joint velocities (n,).
        info : dict
            Dictionary containing:
            - 'position_error': Position error magnitude
            - 'max_joint_vel': Maximum absolute joint velocity (before limiting)
            - 'jacobian': The Jacobian matrix
        """
        cfg = np.asarray(cfg)
        target_position = np.asarray(target_position)

        # Get current pose
        current_pos, current_wxyz = self.get_current_pose(cfg)

        # Compute position error
        pos_error = target_position - current_pos
        pos_error_mag = np.linalg.norm(pos_error)

        # Compute task-space velocity
        if self.position_only:
            v_des = self.position_gain * pos_error
        else:
            if target_wxyz is None:
                target_wxyz = current_wxyz  # Maintain current orientation

            # Ensure quaternions are normalized
            target_wxyz = np.asarray(target_wxyz)
            target_wxyz = target_wxyz / np.linalg.norm(target_wxyz)
            current_wxyz = current_wxyz / np.linalg.norm(current_wxyz)

            # Ensure shortest path (q and -q represent same rotation)
            # If dot product is negative, flip one quaternion
            if np.dot(target_wxyz, current_wxyz) < 0:
                target_wxyz = -target_wxyz

            # Orientation error using quaternion difference
            # q_error = q_target * q_current^-1
            q_current = jaxlie.SO3(jnp.array(current_wxyz))
            q_target = jaxlie.SO3(jnp.array(target_wxyz))
            q_error = q_target @ q_current.inverse()
            omega_error = np.array(q_error.log())

            # Clamp orientation error to prevent large jumps
            omega_mag = np.linalg.norm(omega_error)
            max_omega = 1.0  # rad - maximum orientation error magnitude
            if omega_mag > max_omega:
                omega_error = omega_error * max_omega / omega_mag

            v_des = np.concatenate([
                self.position_gain * pos_error,
                self.orientation_gain * omega_error
            ])

        # Get Jacobian
        jacobian = self.get_jacobian(cfg)

        # Compute pseudo-inverse (with nullspace if enabled)
        J_inv, J_nullspace = self.compute_inverse(
            jacobian, return_nullspace=self.nullspace_enabled
        )

        # Compute primary task joint velocities
        dq_task = J_inv @ v_des
        dq_task = np.asarray(dq_task).flatten()

        # Compute nullspace motion toward home configuration
        dq_nullspace = np.zeros_like(dq_task)
        if self.nullspace_enabled and J_nullspace is not None:
            # Get home configuration (middle of joint range if not set)
            if self._home_cfg is None:
                lower = np.array(self.robot.joints.lower_limits)
                upper = np.array(self.robot.joints.upper_limits)
                self._home_cfg = (lower + upper) / 2.0

            # Nullspace motion: push joints toward home configuration
            # This is the gradient of ||q - q_home||^2
            q_error = cfg - self._home_cfg
            nullspace_velocity = -self.nullspace_gain * q_error

            # Project through nullspace
            dq_nullspace = J_nullspace @ nullspace_velocity
            dq_nullspace = np.asarray(dq_nullspace).flatten()

        # Combined joint velocities
        dq = dq_task + dq_nullspace

        # Track raw magnitude before limiting
        max_joint_vel = np.max(np.abs(dq))

        # Apply velocity limits
        if max_joint_vel > self.max_joint_velocity:
            dq = dq * self.max_joint_velocity / max_joint_vel

        info = {
            'position_error': pos_error_mag,
            'max_joint_vel': max_joint_vel,
            'jacobian': jacobian,
            'nullspace_motion': np.linalg.norm(dq_nullspace),
        }

        return dq, info

    def step(
        self,
        cfg: ArrayLike,
        target_position: np.ndarray,
        target_wxyz: Optional[np.ndarray] = None,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute one control step and return new configuration.

        Parameters
        ----------
        cfg : ArrayLike
            Current joint configuration.
        target_position : np.ndarray
            Target position (3,).
        target_wxyz : np.ndarray, optional
            Target orientation as quaternion (4,).
        dt : float, optional
            Time step. Default is 0.02.

        Returns
        -------
        new_cfg : np.ndarray
            New joint configuration after integration.
        info : dict
            Information dictionary from compute_velocity.
        """
        cfg = np.asarray(cfg)
        dq, info = self.compute_velocity(cfg, target_position, target_wxyz, dt)

        # Integrate
        new_cfg = cfg + dq * dt

        # Clamp to joint limits
        lower = np.array(self.robot.joints.lower_limits)
        upper = np.array(self.robot.joints.upper_limits)
        new_cfg = np.clip(new_cfg, lower, upper)

        return new_cfg, info


def compute_metrics(jacobian: np.ndarray) -> dict:
    """
    Compute manipulability metrics from a Jacobian matrix.

    Parameters
    ----------
    jacobian : np.ndarray
        The Jacobian matrix (m x n).

    Returns
    -------
    dict
        Dictionary containing:
        - 'manipulability': Yoshikawa's manipulability measure
        - 'inverse_condition_number': sigma_min / sigma_max
        - 'singular_values': All singular values
    """
    from jparse_robotics import manipulability_measure, inverse_condition_number

    _, S, _ = np.linalg.svd(jacobian)

    return {
        'manipulability': manipulability_measure(jacobian),
        'inverse_condition_number': inverse_condition_number(jacobian),
        'singular_values': S,
    }
