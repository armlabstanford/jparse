"""
ROS-integrated Robot class.

This module extends the Robot class with ROS-specific functionality
such as loading URDF from the parameter server and publishing
visualization markers to RViz.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..robot import Robot

if TYPE_CHECKING:
    from geometry_msgs.msg import PoseStamped


class ROSRobot(Robot):
    """
    Robot class with ROS integration.

    Extends the base Robot class with ROS-specific functionality:
    - Loading URDF from ROS parameter server
    - Publishing J-PARSE ellipsoid visualizations to RViz

    Parameters
    ----------
    model : pinocchio.Model
        Pinocchio robot model.
    data : pinocchio.Data
        Pinocchio data structure.
    end_frame_id : int
        Frame ID for the end-effector.
    base_frame : str
        Name of the base frame for visualization.
    gamma : float, optional
        J-PARSE singularity threshold. Default is 0.1.

    Examples
    --------
    >>> import rospy
    >>> from jparse_robotics.ros import ROSRobot
    >>>
    >>> rospy.init_node('jparse_node')
    >>> robot = ROSRobot.from_parameter_server("base_link", "ee_link")
    >>>
    >>> q = robot.get_current_joint_state()  # User-implemented
    >>> J_parse = robot.jparse(q)
    """

    def __init__(
        self,
        model,
        data,
        end_frame_id: int,
        base_frame: str,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(model, data, end_frame_id, gamma)
        self._base_frame = base_frame
        self._marker_pub = None

    @classmethod
    def from_parameter_server(
        cls,
        base_link: str,
        end_link: str,
        param_name: str = "/robot_description",
        gamma: float = 0.1,
    ) -> "ROSRobot":
        """
        Create a ROSRobot from URDF on the ROS parameter server.

        Parameters
        ----------
        base_link : str
            Name of the base link.
        end_link : str
            Name of the end-effector link.
        param_name : str, optional
            ROS parameter name for URDF. Default is "/robot_description".
        gamma : float, optional
            J-PARSE singularity threshold. Default is 0.1.

        Returns
        -------
        ROSRobot
            Configured ROSRobot instance.

        Raises
        ------
        ImportError
            If ROS (rospy) or Pinocchio is not available.
        KeyError
            If the URDF parameter is not found.
        """
        try:
            import rospy
        except ImportError:
            raise ImportError(
                "ROS (rospy) is required for this functionality. "
                "Make sure you have ROS installed and sourced."
            )

        try:
            import pinocchio as pin
        except ImportError:
            raise ImportError(
                "Pinocchio is required for URDF support. "
                "Install via: conda install -c conda-forge pinocchio"
            )

        # Get URDF from parameter server
        if not rospy.has_param(param_name):
            raise KeyError(f"URDF parameter '{param_name}' not found on parameter server")

        urdf_string = rospy.get_param(param_name)
        model = pin.buildModelFromXML(urdf_string)
        data = model.createData()

        # Find end-effector frame
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

        return cls(model, data, end_frame_id, base_link, gamma)

    def _ensure_marker_publisher(self) -> None:
        """Initialize the marker publisher if not already done."""
        if self._marker_pub is None:
            import rospy
            from visualization_msgs.msg import MarkerArray

            self._marker_pub = rospy.Publisher(
                "/jparse_ellipsoid_marker", MarkerArray, queue_size=10
            )

    def publish_ellipsoids(
        self,
        q: NDArray[np.floating],
        end_effector_pose: "PoseStamped",
        singular_direction_gain_position: float = 1.0,
        singular_direction_gain_angular: float = 1.0,
    ) -> None:
        """
        Publish J-PARSE ellipsoid visualization markers to RViz.

        This method publishes visualization markers showing:
        - Safety Jacobian ellipsoid (red)
        - Projection Jacobian ellipsoid (blue)
        - Singular direction arrows (red arrows)

        Parameters
        ----------
        q : ndarray
            Current joint configuration.
        end_effector_pose : PoseStamped
            Current end-effector pose for marker positioning.
        singular_direction_gain_position : float, optional
            Position gain for visualization. Default is 1.0.
        singular_direction_gain_angular : float, optional
            Angular gain for visualization. Default is 1.0.
        """
        import rospy
        from geometry_msgs.msg import Point, Quaternion
        from visualization_msgs.msg import Marker, MarkerArray

        self._ensure_marker_publisher()

        # Get position-only Jacobian for visualization
        J = self.jacobian(q)[:3, :]

        # Compute SVD
        U, S, _ = np.linalg.svd(J)
        sigma_max = np.max(S)

        # Compute safety singular values
        S_safety = [
            s if (s / sigma_max) > self._gamma else self._gamma * sigma_max for s in S
        ]

        # Compute projection singular values
        S_proj = [s for s in S if s > self._gamma * sigma_max]

        marker_array = MarkerArray()

        # Safety ellipsoid marker
        safety_marker = Marker()
        safety_marker.header.frame_id = self._base_frame
        safety_marker.header.stamp = rospy.Time.now()
        safety_marker.ns = "J_safety"
        safety_marker.id = 0
        safety_marker.type = Marker.SPHERE
        safety_marker.action = Marker.ADD

        safety_marker.pose.position = end_effector_pose.pose.position
        safety_marker.pose.orientation = self._rotation_to_quaternion(U)

        ellipsoid_scale = 0.25
        safety_marker.scale.x = ellipsoid_scale * max(S_safety[0] if len(S_safety) > 0 else 0.001, 0.001)
        safety_marker.scale.y = ellipsoid_scale * max(S_safety[1] if len(S_safety) > 1 else 0.001, 0.001)
        safety_marker.scale.z = ellipsoid_scale * max(S_safety[2] if len(S_safety) > 2 else 0.001, 0.001)

        safety_marker.color.r = 1.0
        safety_marker.color.g = 0.0
        safety_marker.color.b = 0.0
        safety_marker.color.a = 0.7

        marker_array.markers.append(safety_marker)

        # Projection ellipsoid marker
        if len(S_proj) > 0:
            proj_marker = Marker()
            proj_marker.header.frame_id = self._base_frame
            proj_marker.header.stamp = rospy.Time.now()
            proj_marker.ns = "J_proj"
            proj_marker.id = 1
            proj_marker.type = Marker.SPHERE
            proj_marker.action = Marker.ADD

            proj_marker.pose.position = end_effector_pose.pose.position
            proj_marker.pose.orientation = self._rotation_to_quaternion(U)

            proj_marker.scale.x = ellipsoid_scale * max(S_proj[0] if len(S_proj) > 0 else 0.001, 0.001)
            proj_marker.scale.y = ellipsoid_scale * max(S_proj[1] if len(S_proj) > 1 else 0.001, 0.001)
            proj_marker.scale.z = ellipsoid_scale * max(S_proj[2] if len(S_proj) > 2 else 0.001, 0.001)

            proj_marker.color.r = 0.0
            proj_marker.color.g = 0.0
            proj_marker.color.b = 1.0
            proj_marker.color.a = 0.7

            marker_array.markers.append(proj_marker)

        # Singular direction arrows
        adjusted_cond = [s / sigma_max for s in S]
        for idx, cond in enumerate(adjusted_cond):
            if cond <= self._gamma:
                arrow_marker = Marker()
                arrow_marker.header.frame_id = self._base_frame
                arrow_marker.header.stamp = rospy.Time.now()
                arrow_marker.ns = "J_singular"
                arrow_marker.id = idx + 2
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                arrow_marker.lifetime = rospy.Duration(0.1)

                start_point = Point()
                start_point.x = end_effector_pose.pose.position.x
                start_point.y = end_effector_pose.pose.position.y
                start_point.z = end_effector_pose.pose.position.z

                arrow_scale = 1.0
                phi = cond / self._gamma

                # Determine arrow direction
                ee_pos = np.array([
                    end_effector_pose.pose.position.x,
                    end_effector_pose.pose.position.y,
                    end_effector_pose.pose.position.z,
                ])
                u_dir = U[:, idx]
                if np.dot(ee_pos, u_dir) < 0:
                    arrow_dir = u_dir
                else:
                    arrow_dir = -u_dir

                end_point = Point()
                end_point.x = ee_pos[0] + arrow_scale * arrow_dir[0] * abs(phi)
                end_point.y = ee_pos[1] + arrow_scale * arrow_dir[1] * abs(phi)
                end_point.z = ee_pos[2] + arrow_scale * arrow_dir[2] * abs(phi)

                arrow_marker.points = [start_point, end_point]
                arrow_marker.scale.x = 0.01
                arrow_marker.scale.y = 0.05
                arrow_marker.scale.z = 0.05

                arrow_marker.color.r = 1.0
                arrow_marker.color.g = 0.0
                arrow_marker.color.b = 0.0
                arrow_marker.color.a = 1.0

                marker_array.markers.append(arrow_marker)

        self._marker_pub.publish(marker_array)

    def _rotation_to_quaternion(self, R: NDArray[np.floating]) -> "Quaternion":
        """Convert a 3x3 rotation matrix to a geometry_msgs/Quaternion."""
        from geometry_msgs.msg import Quaternion

        trace = np.trace(R)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        # Normalize
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)

        q = Quaternion()
        q.x = qx / norm
        q.y = qy / norm
        q.z = qz / norm
        q.w = qw / norm
        return q
