"""
Optional ROS integration for jparse-robotics.

This module provides ROS-specific functionality including:
- Loading URDF from ROS parameter server
- Publishing visualization markers to RViz

These imports will only succeed if ROS (rospy) is available.
"""

from __future__ import annotations

try:
    from .ros_robot import ROSRobot

    ROS_AVAILABLE = True
    __all__ = ["ROSRobot", "ROS_AVAILABLE"]
except ImportError:
    ROS_AVAILABLE = False
    ROSRobot = None  # type: ignore
    __all__ = ["ROS_AVAILABLE"]
