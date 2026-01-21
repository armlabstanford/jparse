#!/usr/bin/env python3
"""
LEAP Hand J-PARSE Visualization

Interactive 3D visualization of the LEAP robotic hand controlled using J-PARSE.
Uses viser for web-based visualization (http://localhost:8080).

Features:
- 4 independent finger controllers (Index, Middle, Ring, Thumb)
- Per-finger J-PARSE control with transform gizmos
- Nullspace control to maintain natural finger poses
- Real-time display of manipulability metrics per finger
- Singularity warnings for each finger

Finger Structure (16 DOF total):
- Index:  joints 1, 0, 2, 3  -> fingertip     (4 DOF)
- Middle: joints 5, 4, 6, 7  -> fingertip_2   (4 DOF)
- Ring:   joints 9, 8, 10, 11 -> fingertip_3  (4 DOF)
- Thumb:  joints 12, 13, 14, 15 -> thumb_fingertip (4 DOF)

Requirements:
    pip install jparse-robotics pyroki viser yourdfpy

Run with:
    python examples/pyroki_leap_hand_visualization.py

Then open http://localhost:8080 in your browser.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import trimesh
import viser
import yourdfpy
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from pyroki_jparse_utils import compute_metrics


def icn_to_color(icn: float) -> Tuple[int, int, int]:
    """
    Convert inverse condition number to RGB color.

    ICN ranges from 0 (singular) to 1 (well-conditioned).
    Color goes: Red (danger) -> Yellow (warning) -> Green (safe)
    """
    # Thresholds
    danger_threshold = 0.05
    warning_threshold = 0.15

    if icn < danger_threshold:
        # Red zone - near singularity
        return (255, 50, 50)
    elif icn < warning_threshold:
        # Yellow zone - approaching singularity
        t = (icn - danger_threshold) / (warning_threshold - danger_threshold)
        r = 255
        g = int(50 + t * 155)  # 50 -> 205
        b = 50
        return (r, g, b)
    else:
        # Green zone - well-conditioned
        t = min(1.0, (icn - warning_threshold) / (0.3 - warning_threshold))
        r = int(205 - t * 155)  # 205 -> 50
        g = int(205 + t * 50)   # 205 -> 255
        b = 50
        return (r, g, b)


def icn_to_status(icn: float) -> str:
    """Convert inverse condition number to status string."""
    if icn < 0.05:
        return "SINGULAR!"
    elif icn < 0.10:
        return "Warning"
    elif icn < 0.15:
        return "Caution"
    else:
        return "OK"


def make_progress_bar(value: float, width: int = 10) -> str:
    """Create a text-based progress bar."""
    filled = int(value * width)
    empty = width - filled
    return "[" + "=" * filled + " " * empty + "]"


def compute_ellipsoid_from_jacobian(
    jacobian: np.ndarray,
    scaling: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute manipulability ellipsoid axes and radii from Jacobian.

    The manipulability ellipsoid is defined by J @ J^T.
    Eigendecomposition gives us the principal axes and their lengths.

    Parameters
    ----------
    jacobian : np.ndarray
        Position Jacobian (3 x n).
    scaling : float
        Scaling factor for visualization.

    Returns
    -------
    radii : np.ndarray
        Ellipsoid radii along principal axes (3,).
    rotation_matrix : np.ndarray
        Rotation matrix (3x3) defining ellipsoid orientation.
    """
    # Compute manipulability matrix M = J @ J^T
    M = jacobian @ jacobian.T

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Eigenvalues are squared singular values, radii are sqrt
    # Clamp to avoid numerical issues
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    radii = np.sqrt(eigenvalues) * scaling

    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    return radii, eigenvectors


def update_ellipsoid(
    server: viser.ViserServer,
    name: str,
    position: np.ndarray,
    radii: np.ndarray,
    rotation_matrix: np.ndarray,
    color: Tuple[int, int, int],
    opacity: float = 0.6,
):
    """
    Update or create an ellipsoid visualization in viser.

    Parameters
    ----------
    server : viser.ViserServer
        Viser server instance.
    name : str
        Scene node name for the ellipsoid.
    position : np.ndarray
        Center position (3,).
    radii : np.ndarray
        Ellipsoid radii (3,).
    rotation_matrix : np.ndarray
        Rotation matrix (3x3).
    color : Tuple[int, int, int]
        RGB color.
    opacity : float
        Opacity (0-1).
    """
    # Convert rotation matrix to quaternion (wxyz format)
    rot = R.from_matrix(rotation_matrix)
    quat_xyzw = rot.as_quat()  # scipy returns xyzw
    quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])

    # Create ellipsoid mesh using trimesh
    ellipsoid = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

    # Apply rotation first, then scale
    ellipsoid.vertices = ellipsoid.vertices @ rotation_matrix.T
    ellipsoid.vertices *= radii  # Scale by radii

    # Set vertex colors (RGBA)
    vertex_colors = np.zeros((len(ellipsoid.vertices), 4), dtype=np.uint8)
    vertex_colors[:, 0] = color[0]
    vertex_colors[:, 1] = color[1]
    vertex_colors[:, 2] = color[2]
    vertex_colors[:, 3] = int(opacity * 255)
    ellipsoid.visual.vertex_colors = vertex_colors

    # Add to scene
    server.scene.add_mesh_trimesh(
        name,
        mesh=ellipsoid,
        position=tuple(position),
    )


@dataclass
class FingerConfig:
    """Configuration for a single finger."""
    name: str
    joint_names: List[str]  # Joint names in kinematic order
    tip_link: str  # End-effector link name
    color: Tuple[int, int, int]  # RGB color for visualization


# LEAP Hand finger configurations
FINGER_CONFIGS = {
    "index": FingerConfig(
        name="Index",
        joint_names=["1", "0", "2", "3"],
        tip_link="fingertip",
        color=(255, 100, 100),  # Red
    ),
    "middle": FingerConfig(
        name="Middle",
        joint_names=["5", "4", "6", "7"],
        tip_link="fingertip_2",
        color=(100, 255, 100),  # Green
    ),
    "ring": FingerConfig(
        name="Ring",
        joint_names=["9", "8", "10", "11"],
        tip_link="fingertip_3",
        color=(100, 100, 255),  # Blue
    ),
    "thumb": FingerConfig(
        name="Thumb",
        joint_names=["12", "13", "14", "15"],
        tip_link="thumb_fingertip",
        color=(255, 200, 100),  # Orange
    ),
}


class FingerJParseController:
    """
    J-PARSE controller for a single finger.

    This controller computes joint velocities for a specific finger's joints
    to track a target fingertip position.
    """

    def __init__(
        self,
        robot: pk.Robot,
        config: FingerConfig,
        gamma: float = 0.1,
    ):
        from jparse_robotics import JParseCore

        self.robot = robot
        self.config = config
        self.gamma = gamma

        # Get joint indices in the full robot configuration
        self.joint_indices = []
        for jname in config.joint_names:
            if jname in robot.joints.actuated_names:
                idx = robot.joints.actuated_names.index(jname)
                self.joint_indices.append(idx)
            else:
                raise ValueError(f"Joint {jname} not found in robot actuated joints")

        self.joint_indices = np.array(self.joint_indices)

        # Get tip link index
        self.tip_link_index = robot.links.names.index(config.tip_link)

        # Create J-PARSE solver
        self.jparse = JParseCore(gamma=gamma)

        # Get joint limits from URDF (for this finger's joints)
        self._lower_limits = np.array(robot.joints.lower_limits)[self.joint_indices]
        self._upper_limits = np.array(robot.joints.upper_limits)[self.joint_indices]

        # Control gains
        self.position_gain = 8.0
        self.singular_direction_gain = 2.0
        self.nullspace_gain = 0.3
        self.nullspace_enabled = True

        # Use URDF velocity limit (8.48 rad/s from LEAP hand spec)
        self.max_joint_velocity = 8.0

        # Joint limit avoidance parameters
        self.limit_avoidance_enabled = True
        self.limit_avoidance_gain = 2.0
        self.limit_threshold = 0.1  # Start avoiding when within 0.1 rad of limit

        # Method selection
        self._method = "jparse"
        self._dls_damping = 0.05

        # Home configuration for nullspace (set externally)
        self._home_cfg = None

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, value: str):
        if value not in ["jparse", "pinv", "dls"]:
            raise ValueError(f"method must be 'jparse', 'pinv', or 'dls'")
        self._method = value

    def get_finger_jacobian(self, cfg: np.ndarray) -> np.ndarray:
        """
        Compute position Jacobian for this finger's joints only.

        Parameters
        ----------
        cfg : np.ndarray
            Full robot configuration (all 16 joints).

        Returns
        -------
        np.ndarray
            Position Jacobian (3 x 4) for this finger.
        """
        cfg_jax = jnp.asarray(cfg)

        # Compute full Jacobian via autodiff
        full_jacobian = jax.jacfwd(
            lambda q: jaxlie.SE3(self.robot.forward_kinematics(q)).translation()
        )(cfg_jax)[self.tip_link_index]

        # Extract only columns for this finger's joints
        finger_jacobian = full_jacobian[:, self.joint_indices]

        return np.array(finger_jacobian)

    def get_tip_position(self, cfg: np.ndarray) -> np.ndarray:
        """Get fingertip position."""
        cfg_jax = jnp.asarray(cfg)
        poses = self.robot.forward_kinematics(cfg_jax)
        tip_pose = jaxlie.SE3(poses[self.tip_link_index])
        return np.array(tip_pose.translation().squeeze())

    def compute_inverse(
        self, jacobian: np.ndarray, return_nullspace: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute pseudo-inverse using selected method."""
        J_nullspace = None

        if self._method == "jparse":
            if return_nullspace:
                J_inv, J_nullspace = self.jparse.compute(
                    jacobian,
                    singular_direction_gain_position=self.singular_direction_gain,
                    singular_direction_gain_angular=self.singular_direction_gain,
                    position_dimensions=3,
                    angular_dimensions=0,
                    return_nullspace=True,
                )
            else:
                J_inv = self.jparse.compute(
                    jacobian,
                    singular_direction_gain_position=self.singular_direction_gain,
                    singular_direction_gain_angular=self.singular_direction_gain,
                    position_dimensions=3,
                    angular_dimensions=0,
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

    def step(
        self,
        cfg: np.ndarray,
        target_position: np.ndarray,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute one control step for this finger.

        Parameters
        ----------
        cfg : np.ndarray
            Full robot configuration (all 16 joints).
        target_position : np.ndarray
            Target fingertip position (3,).
        dt : float
            Time step.

        Returns
        -------
        new_cfg : np.ndarray
            Updated full robot configuration.
        info : dict
            Information about the control step.
        """
        cfg = np.asarray(cfg)
        target_position = np.asarray(target_position)

        # Get current tip position
        current_pos = self.get_tip_position(cfg)

        # Compute position error
        pos_error = target_position - current_pos
        pos_error_mag = np.linalg.norm(pos_error)

        # Compute task-space velocity
        v_des = self.position_gain * pos_error

        # Get finger Jacobian
        jacobian = self.get_finger_jacobian(cfg)

        # Compute pseudo-inverse
        J_inv, J_nullspace = self.compute_inverse(
            jacobian, return_nullspace=self.nullspace_enabled
        )

        # Compute primary task joint velocities (for finger joints only)
        dq_finger_task = J_inv @ v_des
        dq_finger_task = np.asarray(dq_finger_task).flatten()

        # Extract current finger joint positions
        cfg_finger = cfg[self.joint_indices]

        # Compute nullspace motion toward home configuration
        dq_finger_nullspace = np.zeros_like(dq_finger_task)
        if self.nullspace_enabled and J_nullspace is not None and self._home_cfg is not None:
            # Extract home config for this finger's joints
            home_finger = self._home_cfg[self.joint_indices]

            # Nullspace motion: push joints toward home
            q_error = cfg_finger - home_finger
            nullspace_velocity = -self.nullspace_gain * q_error

            # Project through nullspace
            dq_finger_nullspace = J_nullspace @ nullspace_velocity
            dq_finger_nullspace = np.asarray(dq_finger_nullspace).flatten()

        # Combined joint velocities for this finger
        dq_finger = dq_finger_task + dq_finger_nullspace

        # Joint limit avoidance - apply repulsive velocity near limits
        if self.limit_avoidance_enabled:
            for i in range(len(cfg_finger)):
                q = cfg_finger[i]
                q_min = self._lower_limits[i]
                q_max = self._upper_limits[i]

                # Distance to limits
                dist_to_lower = q - q_min
                dist_to_upper = q_max - q

                # Repulsive velocity when approaching lower limit
                if dist_to_lower < self.limit_threshold and dq_finger[i] < 0:
                    # Scale down velocity approaching limit
                    scale = dist_to_lower / self.limit_threshold
                    dq_finger[i] *= max(0.0, scale)
                    # Add repulsive component
                    dq_finger[i] += self.limit_avoidance_gain * (self.limit_threshold - dist_to_lower)

                # Repulsive velocity when approaching upper limit
                if dist_to_upper < self.limit_threshold and dq_finger[i] > 0:
                    # Scale down velocity approaching limit
                    scale = dist_to_upper / self.limit_threshold
                    dq_finger[i] *= max(0.0, scale)
                    # Add repulsive component
                    dq_finger[i] -= self.limit_avoidance_gain * (self.limit_threshold - dist_to_upper)

        # Track raw magnitude before limiting
        max_joint_vel = np.max(np.abs(dq_finger))

        # Apply velocity limits (use URDF limit)
        if max_joint_vel > self.max_joint_velocity:
            dq_finger = dq_finger * self.max_joint_velocity / max_joint_vel

        # Create full configuration update
        new_cfg = cfg.copy()
        new_cfg[self.joint_indices] += dq_finger * dt

        # Hard clamp to joint limits as safety (use exact URDF limits)
        lower = np.array(self.robot.joints.lower_limits)
        upper = np.array(self.robot.joints.upper_limits)
        new_cfg = np.clip(new_cfg, lower, upper)

        info = {
            'position_error': pos_error_mag,
            'max_joint_vel': max_joint_vel,
            'jacobian': jacobian,
            'nullspace_motion': np.linalg.norm(dq_finger_nullspace),
        }

        return new_cfg, info


def load_leap_hand():
    """Load LEAP hand URDF with local mesh paths."""
    script_dir = Path(__file__).parent
    leap_dir = script_dir / "leap"
    urdf_path = leap_dir / "leap_hand_right_local.urdf"

    if not urdf_path.exists():
        raise FileNotFoundError(
            f"URDF not found: {urdf_path}\n"
            "Please ensure leap_hand_right_local.urdf exists with local mesh paths."
        )

    # Change to URDF directory so relative mesh paths resolve
    original_dir = os.getcwd()
    os.chdir(leap_dir)
    try:
        urdf = yourdfpy.URDF.load(str(urdf_path))
    finally:
        os.chdir(original_dir)

    return urdf


def main():
    print("=" * 70)
    print("LEAP Hand J-PARSE Visualization")
    print("=" * 70)
    print()

    # Load LEAP hand
    print("Loading LEAP hand URDF...")
    urdf = load_leap_hand()

    # Create pyroki robot
    robot = pk.Robot.from_urdf(urdf)
    print(f"  Actuated joints: {robot.joints.actuated_names}")
    print(f"  Total DOF: {len(robot.joints.actuated_names)}")

    # Create finger controllers
    print("\nCreating finger controllers...")
    controllers: Dict[str, FingerJParseController] = {}
    for key, config in FINGER_CONFIGS.items():
        controllers[key] = FingerJParseController(robot, config, gamma=0.1)
        print(f"  {config.name}: joints {config.joint_names} -> {config.tip_link}")

    # Initial configuration (middle of joint range)
    lower = np.array(robot.joints.lower_limits)
    upper = np.array(robot.joints.upper_limits)
    cfg = (lower + upper) / 2.0

    # Set home configuration for all controllers
    for ctrl in controllers.values():
        ctrl._home_cfg = cfg.copy()

    print()
    print("Starting Viser server...")
    print("Open http://localhost:8080 in your browser")
    print()
    print("Controls:")
    print("  - Drag fingertip gizmos to set target positions")
    print("  - Each finger has independent J-PARSE control")
    print("  - Use GUI controls to adjust parameters")
    print()
    print("Color coding:")
    print("  - Red: Index finger")
    print("  - Green: Middle finger")
    print("  - Blue: Ring finger")
    print("  - Orange: Thumb")
    print("=" * 70)

    # Set up Viser server
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=0.5, height=0.5, cell_size=0.05, plane="xy")

    # Add robot visualization
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Get initial fingertip positions and create gizmos
    targets: Dict[str, viser.TransformControlsHandle] = {}
    for key, ctrl in controllers.items():
        initial_pos = ctrl.get_tip_position(cfg)
        config = FINGER_CONFIGS[key]

        # Add target sphere to show fingertip location
        server.scene.add_icosphere(
            f"/targets/{key}_sphere",
            radius=0.005,
            color=config.color,
        )

        # Add transform gizmo
        targets[key] = server.scene.add_transform_controls(
            f"/targets/{key}",
            scale=0.04,
            position=tuple(initial_pos),
        )

    # -------------------------------------------------------------------------
    # GUI Controls
    # -------------------------------------------------------------------------

    with server.gui.add_folder("Hand Info"):
        server.gui.add_markdown("**LEAP Right Hand - 16 DOF**")
        server.gui.add_markdown("4 fingers with 4 joints each")

    with server.gui.add_folder("IK Method"):
        method_dropdown = server.gui.add_dropdown(
            "Method",
            options=["J-PARSE (Recommended)", "Pseudo-inverse", "Damped LS"],
            initial_value="J-PARSE (Recommended)",
        )

        gamma_slider = server.gui.add_slider(
            "Gamma (J-PARSE)",
            min=0.01,
            max=0.5,
            step=0.01,
            initial_value=0.1,
        )

        singular_gain_slider = server.gui.add_slider(
            "Singular Dir Gain",
            min=0.5,
            max=5.0,
            step=0.1,
            initial_value=2.0,
        )

        dls_damping_slider = server.gui.add_slider(
            "Damping (DLS)",
            min=0.001,
            max=0.2,
            step=0.001,
            initial_value=0.05,
        )

    # Singularity status display (at top level for visibility)
    with server.gui.add_folder("Singularity Status"):
        server.gui.add_markdown("_ICN: 0=singular, 1=optimal_")
        singularity_displays = {}
        for key, config in FINGER_CONFIGS.items():
            singularity_displays[key] = server.gui.add_markdown(
                f"**{config.name}**: `[          ]` OK (0.00)"
            )

    with server.gui.add_folder("Metrics"):
        # Create metric displays for each finger
        finger_metrics = {}
        for key, config in FINGER_CONFIGS.items():
            with server.gui.add_folder(config.name):
                finger_metrics[key] = {
                    'error': server.gui.add_number(
                        "Pos Error (mm)",
                        initial_value=0.0,
                        disabled=True,
                    ),
                    'icn': server.gui.add_number(
                        "Inv Cond Num",
                        initial_value=0.0,
                        disabled=True,
                    ),
                    'manipulability': server.gui.add_number(
                        "Manipulability",
                        initial_value=0.0,
                        disabled=True,
                    ),
                }

    with server.gui.add_folder("Nullspace Control"):
        nullspace_checkbox = server.gui.add_checkbox(
            "Enable Nullspace",
            initial_value=True,
        )

        nullspace_gain_slider = server.gui.add_slider(
            "Nullspace Gain",
            min=0.0,
            max=1.0,
            step=0.05,
            initial_value=0.3,
        )

        server.gui.add_markdown(
            "_Nullspace pulls joints toward neutral pose_"
        )

    with server.gui.add_folder("Control"):
        gain_slider = server.gui.add_slider(
            "Position Gain",
            min=1.0,
            max=15.0,
            step=0.5,
            initial_value=8.0,
        )

        # URDF specifies 8.48 rad/s max velocity
        vel_limit_slider = server.gui.add_slider(
            "Max Joint Vel (rad/s)",
            min=1.0,
            max=8.48,
            step=0.5,
            initial_value=8.0,
        )

        reset_button = server.gui.add_button("Reset to Home")

    with server.gui.add_folder("Joint Limits"):
        limit_avoidance_checkbox = server.gui.add_checkbox(
            "Limit Avoidance",
            initial_value=True,
        )

        limit_avoidance_gain_slider = server.gui.add_slider(
            "Avoidance Gain",
            min=0.5,
            max=5.0,
            step=0.5,
            initial_value=2.0,
        )

        limit_threshold_slider = server.gui.add_slider(
            "Threshold (rad)",
            min=0.02,
            max=0.2,
            step=0.02,
            initial_value=0.1,
        )

        server.gui.add_markdown("_Smoothly avoids joint limits_")

    with server.gui.add_folder("Visualization"):
        show_ellipsoids_checkbox = server.gui.add_checkbox(
            "Show Manipulability Ellipsoids",
            initial_value=True,
        )

        ellipsoid_scale_slider = server.gui.add_slider(
            "Ellipsoid Scale",
            min=0.01,
            max=0.15,
            step=0.01,
            initial_value=0.05,
        )

        ellipsoid_opacity_slider = server.gui.add_slider(
            "Ellipsoid Opacity",
            min=0.1,
            max=1.0,
            step=0.1,
            initial_value=0.6,
        )

        server.gui.add_markdown("_Ellipsoids show motion capability_")

    # Active finger selection
    with server.gui.add_folder("Active Fingers"):
        finger_checkboxes = {}
        for key, config in FINGER_CONFIGS.items():
            finger_checkboxes[key] = server.gui.add_checkbox(
                config.name,
                initial_value=True,
            )

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    @reset_button.on_click
    def on_reset(_):
        nonlocal cfg
        cfg = (lower + upper) / 2.0
        # Update targets to current fingertip positions
        for key, ctrl in controllers.items():
            pos = ctrl.get_tip_position(cfg)
            targets[key].position = tuple(pos)

    # -------------------------------------------------------------------------
    # Main control loop
    # -------------------------------------------------------------------------

    dt = 0.02  # 50 Hz
    last_time = time.time()

    try:
        while True:
            # Timing
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
            last_time = time.time()

            # Update controller parameters from GUI
            method_str = method_dropdown.value
            for ctrl in controllers.values():
                if "J-PARSE" in method_str:
                    ctrl.method = "jparse"
                    ctrl.gamma = gamma_slider.value
                    ctrl.jparse.gamma = gamma_slider.value
                    ctrl.singular_direction_gain = singular_gain_slider.value
                elif "Pseudo-inverse" in method_str:
                    ctrl.method = "pinv"
                else:
                    ctrl.method = "dls"
                    ctrl._dls_damping = dls_damping_slider.value

                ctrl.position_gain = gain_slider.value
                ctrl.max_joint_velocity = vel_limit_slider.value
                ctrl.nullspace_enabled = nullspace_checkbox.value
                ctrl.nullspace_gain = nullspace_gain_slider.value
                ctrl.limit_avoidance_enabled = limit_avoidance_checkbox.value
                ctrl.limit_avoidance_gain = limit_avoidance_gain_slider.value
                ctrl.limit_threshold = limit_threshold_slider.value

            # Control each active finger and track ICN values
            finger_icn_values = {}

            for key, ctrl in controllers.items():
                if not finger_checkboxes[key].value:
                    # Still compute metrics for inactive fingers
                    jacobian = ctrl.get_finger_jacobian(cfg)
                    metrics = compute_metrics(jacobian)
                    finger_icn_values[key] = metrics['inverse_condition_number']
                    finger_metrics[key]['icn'].value = round(metrics['inverse_condition_number'], 4)
                    finger_metrics[key]['manipulability'].value = round(metrics['manipulability'], 6)
                    continue

                # Get target from gizmo
                target_pos = np.array(targets[key].position)

                # Compute control step
                cfg, info = ctrl.step(cfg, target_pos, dt=dt)

                # Update metrics
                metrics = compute_metrics(info['jacobian'])
                icn = metrics['inverse_condition_number']
                finger_icn_values[key] = icn

                finger_metrics[key]['error'].value = round(info['position_error'] * 1000, 2)
                finger_metrics[key]['icn'].value = round(icn, 4)
                finger_metrics[key]['manipulability'].value = round(metrics['manipulability'], 6)

            # Update robot visualization
            urdf_vis.update_cfg(cfg)

            # Update singularity status displays and sphere colors
            for key, ctrl in controllers.items():
                icn = finger_icn_values.get(key, 0.5)
                config = FINGER_CONFIGS[key]

                # Update status display with progress bar
                bar = make_progress_bar(min(1.0, icn / 0.3))  # Normalize to 0.3 for display
                status = icn_to_status(icn)
                singularity_displays[key].content = f"**{config.name}**: `{bar}` {status} ({icn:.3f})"

                # Update target sphere color based on singularity proximity
                pos = np.array(targets[key].position)
                sphere_color = icn_to_color(icn)
                server.scene.add_icosphere(
                    f"/targets/{key}_sphere",
                    radius=0.006,
                    color=sphere_color,
                    position=tuple(pos),
                )

                # Add a warning ring around fingertip when near singularity
                if icn < 0.1:
                    # Show warning ring
                    tip_pos = ctrl.get_tip_position(cfg)
                    server.scene.add_icosphere(
                        f"/warnings/{key}_ring",
                        radius=0.012,
                        color=(255, 0, 0),
                        position=tuple(tip_pos),
                        opacity=0.3,
                    )
                else:
                    # Hide warning ring by making it invisible
                    server.scene.add_icosphere(
                        f"/warnings/{key}_ring",
                        radius=0.001,
                        color=(0, 0, 0),
                        position=(0, 0, -1),  # Move out of view
                        opacity=0.0,
                    )

                # Update manipulability ellipsoid for this finger
                if show_ellipsoids_checkbox.value:
                    # Get fingertip position for ellipsoid center
                    tip_pos = ctrl.get_tip_position(cfg)

                    # Compute ellipsoid from Jacobian
                    jacobian = ctrl.get_finger_jacobian(cfg)
                    radii, rot_matrix = compute_ellipsoid_from_jacobian(
                        jacobian,
                        scaling=ellipsoid_scale_slider.value,
                    )

                    # Update ellipsoid visualization
                    update_ellipsoid(
                        server,
                        f"/ellipsoids/{key}",
                        tip_pos,
                        radii,
                        rot_matrix,
                        color=config.color,
                        opacity=ellipsoid_opacity_slider.value,
                    )
                else:
                    # Hide ellipsoid
                    server.scene.add_icosphere(
                        f"/ellipsoids/{key}",
                        radius=0.001,
                        position=(0, 0, -1),
                        opacity=0.0,
                    )

    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
