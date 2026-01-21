#!/usr/bin/env python3
"""
3D Interactive Visualization with J-PARSE Control

Interactive 3D visualization of robot arms controlled using J-PARSE.
Uses viser for web-based visualization (http://localhost:8080).

Features:
- Multiple robot support: xarm7, panda, iiwa7, ur5, etc.
- Drag transform gizmo to set target pose
- Dropdown to toggle: J-PARSE / Pseudo-inverse / Damped LS
- Slider to adjust J-PARSE gamma parameter
- Singular direction gain for exiting singularities
- Nullspace control to pull joints toward home pose (for redundant robots)
- Real-time display of manipulability, inverse condition number, joint velocities
- Manipulability ellipsoid visualization
- Singularity warning when ICN < 0.1

Requirements:
    pip install jparse-robotics pyroki viser robot-descriptions yourdfpy

Run with:
    python examples/pyroki_3d_visualization.py              # Default: xarm7
    python examples/pyroki_3d_visualization.py --robot panda
    python examples/pyroki_3d_visualization.py --robot iiwa7
    python examples/pyroki_3d_visualization.py --robot ur5

Then open http://localhost:8080 in your browser.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyroki as pk
import viser
import yourdfpy
from viser.extras import ViserUrdf

from pyroki_jparse_utils import JParsePyrokiController, compute_metrics


# Robot configurations
ROBOT_CONFIGS = {
    "xarm7": {
        "urdf_source": "local",  # Use local URDF file
        "urdf_path": "urdf_cache/xarm7.urdf",
        "ee_link": "link_eef",
        "description": "xArm 7-DOF (with gripper)",
    },
    "panda": {
        "urdf_source": "robot_descriptions",
        "urdf_name": "panda_description",
        "ee_link": "panda_hand",
        "description": "Franka Emika Panda 7-DOF",
    },
    "iiwa7": {
        "urdf_source": "robot_descriptions",
        "urdf_name": "iiwa7_description",
        "ee_link": "iiwa_link_ee",
        "description": "KUKA LBR iiwa 7-DOF",
    },
    "iiwa14": {
        "urdf_source": "robot_descriptions",
        "urdf_name": "iiwa14_description",
        "ee_link": "iiwa_link_ee",
        "description": "KUKA LBR iiwa 14kg 7-DOF",
    },
    "ur5": {
        "urdf_source": "robot_descriptions",
        "urdf_name": "ur5_description",
        "ee_link": "ee_link",
        "description": "Universal Robots UR5 6-DOF",
    },
    "ur10": {
        "urdf_source": "robot_descriptions",
        "urdf_name": "ur10_description",
        "ee_link": "ee_link",
        "description": "Universal Robots UR10 6-DOF",
    },
}


def load_robot(robot_key: str):
    """Load robot URDF and return (urdf, ee_link_name)."""
    if robot_key not in ROBOT_CONFIGS:
        raise ValueError(f"Unknown robot: {robot_key}. Available: {list(ROBOT_CONFIGS.keys())}")

    config = ROBOT_CONFIGS[robot_key]

    if config["urdf_source"] == "local":
        # Load from local file - need to change to urdf directory for mesh paths
        script_dir = Path(__file__).parent.parent
        urdf_dir = script_dir / "urdf_cache"
        urdf_path = urdf_dir / "xarm7.urdf"
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        # Change to urdf directory so relative mesh paths resolve
        original_dir = os.getcwd()
        os.chdir(urdf_dir)
        try:
            urdf = yourdfpy.URDF.load(str(urdf_path))
        finally:
            os.chdir(original_dir)
    else:
        # Load from robot_descriptions
        from robot_descriptions.loaders.yourdfpy import load_robot_description
        urdf = load_robot_description(config["urdf_name"])

    return urdf, config["ee_link"], config["description"]


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="3D J-PARSE Visualization")
    parser.add_argument(
        "--robot", "-r",
        type=str,
        default="xarm7",
        choices=list(ROBOT_CONFIGS.keys()),
        help="Robot to visualize (default: xarm7)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available robots and exit"
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable robots:")
        print("-" * 50)
        for key, cfg in ROBOT_CONFIGS.items():
            print(f"  {key:10s} - {cfg['description']}")
        print()
        return

    print("=" * 70)
    print("3D Interactive Visualization with J-PARSE Control")
    print("=" * 70)
    print()

    # Load robot
    print(f"Loading robot: {args.robot}...")
    urdf, target_link_name, robot_desc = load_robot(args.robot)
    print(f"  {robot_desc}")
    print(f"  End-effector link: {target_link_name}")

    # Create pyroki robot
    robot = pk.Robot.from_urdf(urdf)
    print(f"  Actuated joints: {len(robot.joints.actuated_names)}")

    print()
    print("Starting Viser server...")
    print("Open http://localhost:8080 in your browser")
    print()
    print("Controls:")
    print("  - Drag the transform gizmo to set target position")
    print("  - Use dropdown to switch between J-PARSE / Pinv / DLS")
    print("  - Adjust gamma slider to tune J-PARSE sensitivity")
    print("  - Adjust 'Singular Dir Gain' to help exit singularities")
    print()
    print("To observe J-PARSE advantage:")
    print("  1. Drag target near workspace boundary (arm stretched out)")
    print("  2. Watch 'Max Joint Vel' - pinv explodes near singularities!")
    print("  3. Switch to J-PARSE to see bounded velocities")
    print()
    print("To exit singularities:")
    print("  - Increase 'Singular Dir Gain (Exit)' slider (try 2.0 - 4.0)")
    print("  - This allows controlled motion along singular directions")
    print()
    print("Nullspace control (7-DOF robots):")
    print("  - Enabled by default - pulls joints toward home pose")
    print("  - Adjust 'Nullspace Gain' to change strength")
    print("  - Motion is in nullspace, so it doesn't affect end-effector!")
    print("=" * 70)

    # Create J-PARSE controller
    controller = JParsePyrokiController(
        robot,
        target_link_name,
        gamma=0.1,
        position_only=True,
    )

    # Initial configuration (middle of joint range)
    lower = np.array(robot.joints.lower_limits)
    upper = np.array(robot.joints.upper_limits)
    cfg = (lower + upper) / 2.0

    # Set up Viser server
    server = viser.ViserServer()

    # Set Z-up coordinate system (standard for robotics)
    server.scene.set_up_direction("+z")

    # Add ground plane (xy plane is horizontal when z is up)
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1, plane="xy")

    # Add robot visualization
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Get initial end-effector pose
    initial_pos, initial_wxyz = controller.get_current_pose(cfg)

    # Add transform gizmo for target
    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.15,
        position=tuple(initial_pos),
        wxyz=tuple(initial_wxyz),
    )

    # Add manipulability ellipsoid (solid, not wireframe)
    manip_ellipse = pk.viewer.ManipulabilityEllipse(
        server,
        robot,
        root_node_name="/manipulability",
        target_link_name=target_link_name,
        scaling_factor=0.15,
        visible=True,
        wireframe=False,
        color=(200, 100, 255),
    )

    # -------------------------------------------------------------------------
    # GUI Controls
    # -------------------------------------------------------------------------

    # Robot info
    with server.gui.add_folder("Robot"):
        server.gui.add_markdown(f"**{robot_desc}**")
        server.gui.add_markdown(f"Joints: {len(robot.joints.actuated_names)}")

    # Folder for controls
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

        # Singular direction gains - KEY for exiting singularities!
        singular_gain_slider = server.gui.add_slider(
            "Singular Dir Gain (Exit)",
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

    with server.gui.add_folder("Metrics"):
        manip_handle = server.gui.add_number(
            "Manipulability",
            initial_value=0.0,
            disabled=True,
        )

        icn_handle = server.gui.add_number(
            "Inv. Cond. Number",
            initial_value=0.0,
            disabled=True,
        )

        max_vel_handle = server.gui.add_number(
            "Max Joint Vel (rad/s)",
            initial_value=0.0,
            disabled=True,
        )

        pos_error_handle = server.gui.add_number(
            "Position Error (m)",
            initial_value=0.0,
            disabled=True,
        )

        # Singularity warning
        singularity_warning = server.gui.add_markdown(
            "_Status: Normal_"
        )

    with server.gui.add_folder("Nullspace Control"):
        nullspace_checkbox = server.gui.add_checkbox(
            "Enable Nullspace Motion",
            initial_value=True,
        )

        nullspace_gain_slider = server.gui.add_slider(
            "Nullspace Gain",
            min=0.0,
            max=2.0,
            step=0.05,
            initial_value=0.5,
        )

        nullspace_motion_handle = server.gui.add_number(
            "Nullspace Motion",
            initial_value=0.0,
            disabled=True,
        )

        server.gui.add_markdown(
            "_Nullspace pulls joints toward home pose without affecting end-effector_"
        )

    with server.gui.add_folder("Control"):
        gain_slider = server.gui.add_slider(
            "Position Gain",
            min=0.5,
            max=10.0,
            step=0.1,
            initial_value=5.0,
        )

        orientation_gain_slider = server.gui.add_slider(
            "Orientation Gain",
            min=0.1,
            max=3.0,
            step=0.1,
            initial_value=1.0,
        )

        vel_limit_slider = server.gui.add_slider(
            "Max Joint Vel Limit",
            min=0.5,
            max=5.0,
            step=0.1,
            initial_value=2.0,
        )

        show_ellipsoid = server.gui.add_checkbox(
            "Show Manipulability Ellipsoid",
            initial_value=True,
        )

        reset_button = server.gui.add_button("Reset to Home")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    @reset_button.on_click
    def on_reset(_):
        nonlocal cfg
        cfg = (lower + upper) / 2.0
        # Update target to current EE position
        pos, wxyz = controller.get_current_pose(cfg)
        ik_target.position = tuple(pos)
        ik_target.wxyz = tuple(wxyz)

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
            if "J-PARSE" in method_str:
                controller.method = "jparse"
                controller.gamma = gamma_slider.value
                controller.jparse.gamma = gamma_slider.value
                # Singular direction gain - helps exit singularities!
                controller.singular_direction_gain_position = singular_gain_slider.value
                controller.singular_direction_gain_angular = singular_gain_slider.value
            elif "Pseudo-inverse" in method_str:
                controller.method = "pinv"
            else:
                controller.method = "dls"
                controller._dls_damping = dls_damping_slider.value

            controller.position_gain = gain_slider.value
            controller.orientation_gain = orientation_gain_slider.value
            controller.max_joint_velocity = vel_limit_slider.value

            # Update nullspace settings
            controller.nullspace_enabled = nullspace_checkbox.value
            controller.nullspace_gain = nullspace_gain_slider.value

            # Get target from gizmo (position only for now)
            target_pos = np.array(ik_target.position)
            target_wxyz = None

            # Compute control step
            cfg, info = controller.step(cfg, target_pos, target_wxyz=target_wxyz, dt=dt)

            # Update robot visualization
            urdf_vis.update_cfg(cfg)

            # Update manipulability ellipsoid
            if show_ellipsoid.value:
                manip_ellipse.set_visibility(True)
                manip_ellipse.update(cfg)
            else:
                manip_ellipse.set_visibility(False)

            # Compute metrics
            metrics = compute_metrics(info['jacobian'])

            # Update GUI displays
            manip_handle.value = round(metrics['manipulability'], 4)
            icn_handle.value = round(metrics['inverse_condition_number'], 4)
            max_vel_handle.value = round(info['max_joint_vel'], 3)
            pos_error_handle.value = round(info['position_error'], 4)
            nullspace_motion_handle.value = round(info.get('nullspace_motion', 0.0), 4)

            # Update singularity warning
            icn = metrics['inverse_condition_number']
            if icn < 0.05:
                singularity_warning.content = "**WARNING: NEAR SINGULARITY!**"
            elif icn < 0.1:
                singularity_warning.content = "_Approaching singularity_"
            else:
                singularity_warning.content = "_Status: Normal_"

    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
