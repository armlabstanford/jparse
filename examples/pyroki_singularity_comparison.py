#!/usr/bin/env python3
"""
Singularity Comparison Demo

Automated trajectory that passes through singularities, comparing
joint velocity behavior between J-PARSE, Pseudo-inverse, and Damped LS.

This script demonstrates how J-PARSE maintains bounded velocities
near singularities while standard pseudo-inverse explodes.

Requirements:
    pip install jparse-robotics pyroki matplotlib

Run with:
    python examples/pyroki_singularity_comparison.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description

from pyroki_jparse_utils import JParsePyrokiController, compute_metrics


def run_trajectory(
    controller: JParsePyrokiController,
    trajectory: np.ndarray,
    method: str,
    dt: float = 0.02,
) -> dict:
    """
    Run a trajectory and collect metrics.

    Parameters
    ----------
    controller : JParsePyrokiController
        The controller to use.
    trajectory : np.ndarray
        Target positions (N, 3).
    method : str
        IK method to use ('jparse', 'pinv', 'dls').
    dt : float
        Time step.

    Returns
    -------
    dict
        Dictionary with trajectory results.
    """
    controller.method = method

    # Reset to initial configuration
    lower = np.array(controller.robot.joints.lower_limits)
    upper = np.array(controller.robot.joints.upper_limits)
    cfg = (lower + upper) / 2.0

    positions = []
    max_vels = []
    icns = []
    manipulabilities = []
    cfgs = []

    for target in trajectory:
        # Run several steps to allow convergence
        for _ in range(5):
            cfg, info = controller.step(cfg, target, dt=dt)

        pos, _ = controller.get_current_pose(cfg)
        metrics = compute_metrics(info['jacobian'])

        positions.append(pos.copy())
        max_vels.append(info['max_joint_vel'])
        icns.append(metrics['inverse_condition_number'])
        manipulabilities.append(metrics['manipulability'])
        cfgs.append(cfg.copy())

    return {
        'positions': np.array(positions),
        'max_vels': np.array(max_vels),
        'icns': np.array(icns),
        'manipulabilities': np.array(manipulabilities),
        'cfgs': np.array(cfgs),
    }


def main():
    print("=" * 70)
    print("Singularity Comparison: J-PARSE vs Pseudo-inverse vs Damped LS")
    print("=" * 70)
    print()

    # Load Panda robot
    print("Loading Panda robot...")
    urdf = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(urdf)

    # Create controller
    controller = JParsePyrokiController(
        robot,
        "panda_hand",
        gamma=0.1,
        position_only=True,
    )
    controller.max_joint_velocity = 10.0  # High limit to show raw velocities

    # Get initial pose
    lower = np.array(robot.joints.lower_limits)
    upper = np.array(robot.joints.upper_limits)
    cfg = (lower + upper) / 2.0
    start_pos, _ = controller.get_current_pose(cfg)

    # Create trajectory that goes toward workspace boundary (singularity)
    # Moving outward along x-axis
    print("Creating trajectory toward singularity...")
    n_points = 50

    # Start from current position, extend toward arm limit
    trajectory = np.zeros((n_points, 3))
    for i in range(n_points):
        t = i / (n_points - 1)
        trajectory[i] = start_pos + np.array([
            t * 0.35,  # Move outward (toward singularity)
            0.0,
            -t * 0.1,  # Slight downward motion
        ])

    print(f"Trajectory from {trajectory[0]} to {trajectory[-1]}")
    print()

    # Run with each method
    methods = {
        'J-PARSE': 'jparse',
        'Pseudo-inverse': 'pinv',
        'Damped LS': 'dls',
    }

    results = {}
    for name, method in methods.items():
        print(f"Running trajectory with {name}...")
        results[name] = run_trajectory(controller, trajectory, method)
        print(f"  Max joint velocity: {results[name]['max_vels'].max():.2f} rad/s")
        print(f"  Min ICN: {results[name]['icns'].min():.4f}")

    # Plotting
    print()
    print("Creating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Max joint velocity over trajectory
    ax1 = axes[0, 0]
    for name in methods.keys():
        ax1.plot(results[name]['max_vels'], label=name, linewidth=2)
    ax1.set_xlabel('Trajectory Step')
    ax1.set_ylabel('Max Joint Velocity (rad/s)')
    ax1.set_title('Joint Velocities Near Singularity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Inverse condition number
    ax2 = axes[0, 1]
    for name in methods.keys():
        ax2.plot(results[name]['icns'], label=name, linewidth=2)
    ax2.axhline(y=0.1, color='r', linestyle='--', label='Singularity threshold')
    ax2.set_xlabel('Trajectory Step')
    ax2.set_ylabel('Inverse Condition Number')
    ax2.set_title('Singularity Proximity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: End-effector trajectory (X-Z plane)
    ax3 = axes[1, 0]
    for name in methods.keys():
        pos = results[name]['positions']
        ax3.plot(pos[:, 0], pos[:, 2], label=name, linewidth=2)
    ax3.plot(trajectory[:, 0], trajectory[:, 2], 'k--', alpha=0.5, label='Target')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('End-Effector Trajectory (X-Z plane)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    # Plot 4: Manipulability
    ax4 = axes[1, 1]
    for name in methods.keys():
        ax4.plot(results[name]['manipulabilities'], label=name, linewidth=2)
    ax4.set_xlabel('Trajectory Step')
    ax4.set_ylabel('Manipulability')
    ax4.set_title('Yoshikawa Manipulability Index')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = 'examples/singularity_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved figure to: {output_path}")

    plt.show()

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Method':<20} {'Max Vel (rad/s)':<18} {'Min ICN':<12}")
    print("-" * 50)
    for name in methods.keys():
        max_vel = results[name]['max_vels'].max()
        min_icn = results[name]['icns'].min()
        print(f"{name:<20} {max_vel:<18.2f} {min_icn:<12.4f}")
    print()
    print("Note: Lower max velocity near singularities indicates better handling.")
    print("J-PARSE should show bounded velocities even when ICN drops.")


if __name__ == "__main__":
    main()
