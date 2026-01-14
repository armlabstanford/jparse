"""
High-level Robot class with URDF support via Pinocchio.

This module provides a convenient interface for using J-PARSE with real robot
models defined in URDF format.
"""

from __future__ import annotations

import os
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .core import JParseCore, inverse_condition_number, manipulability_measure


class Robot:
    """
    High-level robot interface for J-PARSE computations.

    This class combines Pinocchio kinematics with the J-PARSE algorithm to
    provide singularity-aware inverse kinematics for robots defined in URDF.

    Parameters
    ----------
    model : pinocchio.Model
        Pinocchio robot model.
    data : pinocchio.Data
        Pinocchio data structure.
    end_frame_id : int
        Frame ID for the end-effector.
    gamma : float, optional
        J-PARSE singularity threshold. Default is 0.1.

    Examples
    --------
    >>> from jparse_robotics import Robot
    >>> import numpy as np
    >>>
    >>> # Load robot from URDF
    >>> robot = Robot.from_urdf("robot.urdf", "base_link", "ee_link")
    >>>
    >>> # Get J-PARSE pseudo-inverse at a configuration
    >>> q = np.zeros(robot.num_joints)
    >>> J_parse = robot.jparse(q)
    >>>
    >>> # Velocity control
    >>> desired_velocity = np.array([0.1, 0, 0, 0, 0, 0])  # [vx, vy, vz, wx, wy, wz]
    >>> joint_velocities = J_parse @ desired_velocity

    See Also
    --------
    JParseCore : Pure J-PARSE algorithm without kinematics dependencies.
    """

    def __init__(
        self,
        model,  # pinocchio.Model
        data,  # pinocchio.Data
        end_frame_id: int,
        gamma: float = 0.1,
    ) -> None:
        self._model = model
        self._data = data
        self._end_frame_id = end_frame_id
        self._jparse_core = JParseCore(gamma=gamma)
        self._gamma = gamma

    @classmethod
    def from_urdf(
        cls,
        urdf: str,
        base_link: str,
        end_link: str,
        gamma: float = 0.1,
    ) -> "Robot":
        """
        Create a Robot instance from a URDF file or XML string.

        Parameters
        ----------
        urdf : str
            Path to URDF file or URDF XML string.
        base_link : str
            Name of the base link (fixed frame).
        end_link : str
            Name of the end-effector link.
        gamma : float, optional
            J-PARSE singularity threshold. Default is 0.1.

        Returns
        -------
        Robot
            Configured Robot instance.

        Raises
        ------
        ImportError
            If Pinocchio is not installed.
        ValueError
            If the specified links are not found in the URDF.

        Examples
        --------
        >>> # From file path
        >>> robot = Robot.from_urdf("/path/to/robot.urdf", "base", "tool0")
        >>>
        >>> # From XML string
        >>> urdf_xml = '''<?xml version="1.0"?>
        ... <robot name="simple_arm">
        ...   <!-- URDF content -->
        ... </robot>'''
        >>> robot = Robot.from_urdf(urdf_xml, "base", "end")
        """
        try:
            import pinocchio as pin
        except ImportError:
            raise ImportError(
                "Pinocchio is required for URDF support. "
                "Install via: conda install -c conda-forge pinocchio\n"
                "Or for pure algorithm usage without URDF, use JParseCore instead."
            )

        # Determine if urdf is a file path or XML string
        is_path = os.path.isfile(urdf) or urdf.endswith(".urdf")

        if is_path:
            if not os.path.isfile(urdf):
                raise FileNotFoundError(f"URDF file not found: {urdf}")
            model = pin.buildModelFromUrdf(urdf)
        else:
            # Assume it's an XML string
            model = pin.buildModelFromXML(urdf)

        data = model.createData()

        # Find the end-effector frame
        end_frame_id = None
        for i, frame in enumerate(model.frames):
            if frame.name == end_link:
                end_frame_id = i
                break

        if end_frame_id is None:
            available_frames = [f.name for f in model.frames]
            raise ValueError(
                f"End link '{end_link}' not found in URDF. "
                f"Available frames: {available_frames}"
            )

        return cls(model, data, end_frame_id, gamma)

    @property
    def num_joints(self) -> int:
        """Number of actuated joints in the robot."""
        return self._model.nq

    @property
    def num_velocities(self) -> int:
        """Number of velocity degrees of freedom."""
        return self._model.nv

    @property
    def gamma(self) -> float:
        """J-PARSE singularity threshold."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        if not 0 < value < 1:
            raise ValueError(f"gamma must be in (0, 1), got {value}")
        self._gamma = value
        self._jparse_core = JParseCore(gamma=value)

    def jacobian(self, q: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the geometric Jacobian at configuration q.

        Parameters
        ----------
        q : ndarray
            Joint configuration (length num_joints).

        Returns
        -------
        ndarray
            6 x num_velocities Jacobian matrix.
            Rows 0-2: linear velocity components (vx, vy, vz)
            Rows 3-5: angular velocity components (wx, wy, wz)
        """
        import pinocchio as pin

        q = np.asarray(q).flatten()
        pin.computeJointJacobians(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)

        J = pin.getFrameJacobian(
            self._model,
            self._data,
            self._end_frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return J

    def forward_kinematics(
        self, q: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute forward kinematics at configuration q.

        Parameters
        ----------
        q : ndarray
            Joint configuration (length num_joints).

        Returns
        -------
        position : ndarray
            End-effector position (x, y, z).
        rotation : ndarray
            End-effector rotation matrix (3x3).
        """
        import pinocchio as pin

        q = np.asarray(q).flatten()
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)

        oMf = self._data.oMf[self._end_frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def jparse(
        self,
        q: NDArray[np.floating],
        position_only: bool = False,
        return_nullspace: bool = False,
        singular_direction_gain_position: float = 1.0,
        singular_direction_gain_angular: float = 1.0,
    ) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], NDArray[np.floating]]]:
        """
        Compute J-PARSE pseudo-inverse at configuration q.

        Parameters
        ----------
        q : ndarray
            Joint configuration (length num_joints).
        position_only : bool, optional
            If True, only use position rows of Jacobian (3xn instead of 6xn).
            Default is False.
        return_nullspace : bool, optional
            If True, also return nullspace projection matrix. Default is False.
        singular_direction_gain_position : float, optional
            Gain for position singular directions. Default is 1.0.
        singular_direction_gain_angular : float, optional
            Gain for angular singular directions. Default is 1.0.

        Returns
        -------
        J_parse : ndarray
            J-PARSE pseudo-inverse matrix (num_velocities x 6 or num_velocities x 3).
        nullspace : ndarray, optional
            Nullspace projection matrix (only if return_nullspace=True).
        """
        J = self.jacobian(q)

        if position_only:
            J = J[:3, :]
            position_dims = 3
            angular_dims = None
        else:
            position_dims = 3
            angular_dims = 3

        return self._jparse_core.compute(
            J,
            singular_direction_gain_position=singular_direction_gain_position,
            singular_direction_gain_angular=singular_direction_gain_angular,
            position_dimensions=position_dims,
            angular_dimensions=angular_dims,
            return_nullspace=return_nullspace,
        )

    def pinv(
        self, q: NDArray[np.floating], position_only: bool = False
    ) -> NDArray[np.floating]:
        """
        Compute standard pseudo-inverse at configuration q.

        This is provided for comparison with J-PARSE.

        Parameters
        ----------
        q : ndarray
            Joint configuration.
        position_only : bool, optional
            If True, only use position rows of Jacobian.

        Returns
        -------
        ndarray
            Pseudo-inverse of the Jacobian.
        """
        J = self.jacobian(q)
        if position_only:
            J = J[:3, :]
        return self._jparse_core.pinv(J)

    def damped_least_squares(
        self,
        q: NDArray[np.floating],
        damping: float = 0.01,
        position_only: bool = False,
    ) -> NDArray[np.floating]:
        """
        Compute damped least squares pseudo-inverse at configuration q.

        This is provided for comparison with J-PARSE.

        Parameters
        ----------
        q : ndarray
            Joint configuration.
        damping : float, optional
            Damping factor. Default is 0.01.
        position_only : bool, optional
            If True, only use position rows of Jacobian.

        Returns
        -------
        ndarray
            Damped least squares pseudo-inverse.
        """
        J = self.jacobian(q)
        if position_only:
            J = J[:3, :]
        return self._jparse_core.damped_least_squares(J, damping)

    def manipulability(self, q: NDArray[np.floating]) -> float:
        """
        Compute Yoshikawa's manipulability measure at configuration q.

        Parameters
        ----------
        q : ndarray
            Joint configuration.

        Returns
        -------
        float
            Manipulability measure.
        """
        J = self.jacobian(q)
        return manipulability_measure(J)

    def inverse_condition_number(self, q: NDArray[np.floating]) -> float:
        """
        Compute inverse condition number at configuration q.

        Parameters
        ----------
        q : ndarray
            Joint configuration.

        Returns
        -------
        float
            Inverse condition number (sigma_min / sigma_max).
        """
        J = self.jacobian(q)
        return inverse_condition_number(J)
