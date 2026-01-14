#!/usr/bin/env python3
"""
Using J-PARSE with a URDF robot model.

This example demonstrates how to use the Robot class with a URDF file.
The Robot class handles kinematics internally using Pinocchio.

Requirements:
    pip install jparse-robotics
    conda install -c conda-forge pinocchio

Or if using pip only (limited platform support):
    pip install jparse-robotics pin
"""

import numpy as np

# Check for Pinocchio
try:
    import pinocchio  # noqa: F401

    PINOCCHIO_AVAILABLE = True
except ImportError:
    PINOCCHIO_AVAILABLE = False
    print("WARNING: Pinocchio not available.")
    print("Install via: conda install -c conda-forge pinocchio")
    print()


def create_sample_urdf(path: str):
    """Create a sample 7-DOF arm URDF for demonstration."""
    urdf_content = """<?xml version="1.0"?>
<robot name="demo_arm">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
"""

    # Create 7 links and joints
    joint_axes = [
        "0 0 1",  # Z
        "0 1 0",  # Y
        "0 0 1",  # Z
        "0 1 0",  # Y
        "0 0 1",  # Z
        "0 1 0",  # Y
        "0 0 1",  # Z
    ]
    link_lengths = [0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0]  # Alternating

    prev_link = "base_link"
    for i in range(7):
        link_name = f"link{i+1}"
        joint_name = f"joint{i+1}"

        urdf_content += f"""
  <link name="{link_name}">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="{link_lengths[i]/2} 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="{joint_name}" type="revolute">
    <parent link="{prev_link}"/>
    <child link="{link_name}"/>
    <origin xyz="{link_lengths[i]} 0 0" rpy="0 0 0"/>
    <axis xyz="{joint_axes[i]}"/>
    <limit lower="-2.96" upper="2.96" effort="100" velocity="2"/>
  </joint>
"""
        prev_link = link_name

    # End-effector
    urdf_content += """
  <link name="ee_link">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0.05 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="ee_joint" type="fixed">
    <parent link="link7"/>
    <child link="ee_link"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""

    with open(path, "w") as f:
        f.write(urdf_content)

    return path


def main():
    if not PINOCCHIO_AVAILABLE:
        print("This example requires Pinocchio. Exiting.")
        return

    import jparse_robotics as jparse

    # Create a temporary URDF file
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        urdf_path = os.path.join(tmpdir, "demo_arm.urdf")
        create_sample_urdf(urdf_path)

        print("=" * 60)
        print("J-PARSE Demo: 7-DOF Arm with URDF")
        print("=" * 60)

        # Load robot from URDF
        robot = jparse.Robot.from_urdf(
            urdf_path,
            base_link="base_link",
            end_link="ee_link",
            gamma=0.1,
        )

        print(f"Loaded robot with {robot.num_joints} joints")
        print(f"J-PARSE gamma: {robot.gamma}")
        print()

        # Initial configuration (all zeros)
        q = np.zeros(robot.num_joints)

        # Get forward kinematics
        pos, rot = robot.forward_kinematics(q)
        print("Initial configuration (all zeros):")
        print(f"  End-effector position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"  Manipulability: {robot.manipulability(q):.4f}")
        print(f"  Inverse condition number: {robot.inverse_condition_number(q):.4f}")
        print()

        # Get Jacobian
        J = robot.jacobian(q)
        print(f"Jacobian shape: {J.shape}")
        print()

        # Get J-PARSE pseudo-inverse
        J_parse = robot.jparse(q)
        print(f"J-PARSE shape: {J_parse.shape}")
        print()

        # Velocity control example
        print("Velocity control example:")
        print("-" * 40)

        # Desired Cartesian velocity (move in x direction)
        v_des = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        print(f"Desired velocity: {v_des[:3]} (linear), {v_des[3:]} (angular)")

        # Compute joint velocities
        dq = J_parse @ v_des
        print(f"Joint velocities: {dq}")
        print()

        # Compare methods
        print("Comparison of IK methods:")
        print("-" * 40)

        J_pinv = robot.pinv(q)
        J_dls = robot.damped_least_squares(q, damping=0.1)

        dq_pinv = J_pinv @ v_des
        dq_dls = J_dls @ v_des

        print(f"Standard pinv: max |dq| = {np.max(np.abs(dq_pinv)):.4f}")
        print(f"J-PARSE:       max |dq| = {np.max(np.abs(dq)):.4f}")
        print(f"DLS (λ=0.1):   max |dq| = {np.max(np.abs(dq_dls)):.4f}")
        print()

        # Nullspace motion example
        print("Nullspace motion example:")
        print("-" * 40)

        J_parse, nullspace = robot.jparse(q, return_nullspace=True)
        print(f"Nullspace matrix shape: {nullspace.shape}")

        # Secondary objective: move toward zero configuration
        q_desired = np.zeros(robot.num_joints)
        q_secondary = nullspace @ (q_desired - q)
        print(f"Nullspace motion: {q_secondary}")
        print()

        # Position-only control
        print("Position-only control:")
        print("-" * 40)

        J_parse_pos = robot.jparse(q, position_only=True)
        print(f"Position-only J-PARSE shape: {J_parse_pos.shape}")

        v_des_pos = np.array([0.1, 0.05, 0.0])  # Only linear velocity
        dq_pos = J_parse_pos @ v_des_pos
        print(f"Joint velocities for position control: {dq_pos}")
        print()

        # Trajectory tracking simulation
        print("Trajectory tracking simulation:")
        print("-" * 40)

        dt = 0.01
        num_steps = 100
        q = np.zeros(robot.num_joints)

        # Track a line in x direction
        k = 2.0
        target_vel = np.array([0.1, 0.0, 0.0])

        for i in range(num_steps):
            pos, _ = robot.forward_kinematics(q)
            J_parse = robot.jparse(q, position_only=True)

            # Simple velocity control
            dq = J_parse @ target_vel
            q = q + dq * dt

        pos_final, _ = robot.forward_kinematics(q)
        print(f"Initial position: [{0:.3f}, {0:.3f}, {0:.3f}]")
        print(f"Final position:   [{pos_final[0]:.3f}, {pos_final[1]:.3f}, {pos_final[2]:.3f}]")
        print(f"Movement in x:    {pos_final[0] - 0:.3f}m")

        print()
        print("Demo complete!")


if __name__ == "__main__":
    main()
