#!/usr/bin/env python3
"""
2D Arm Visualization with J-PARSE Control

Interactive visualization of a 2-link planar arm controlled using J-PARSE.
- Click anywhere to set a target position
- The arm will track to that position using J-PARSE inverse kinematics
- Shows manipulability ellipsoid and singularity metrics in real-time

Requirements: pip install matplotlib numpy jparse-robotics
Run with: python visualization_2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.animation import FuncAnimation
import jparse_robotics as jparse


class PlanarArm2D:
    """2-link planar arm with J-PARSE control and visualization."""

    def __init__(self, l1: float = 1.0, l2: float = 1.0, gamma: float = 0.1):
        self.l1 = l1
        self.l2 = l2
        self.solver = jparse.JParseCore(gamma=gamma)

        # Joint angles [rad]
        self.q = np.array([np.pi/4, np.pi/4])

        # Target position (initially at current end-effector)
        self.target = self.forward_kinematics()

        # Control gain
        self.k = 3.0

        # Toggle between J-PARSE and standard pinv
        self.use_jparse = True

        # Track joint velocity magnitude for comparison
        self.last_dq_magnitude = 0.0

    def forward_kinematics(self) -> np.ndarray:
        """Compute end-effector position."""
        x = self.l1 * np.cos(self.q[0]) + self.l2 * np.cos(self.q[0] + self.q[1])
        y = self.l1 * np.sin(self.q[0]) + self.l2 * np.sin(self.q[0] + self.q[1])
        return np.array([x, y])

    def get_joint_positions(self) -> tuple:
        """Get positions of base, elbow, and end-effector."""
        base = np.array([0.0, 0.0])
        elbow = np.array([
            self.l1 * np.cos(self.q[0]),
            self.l1 * np.sin(self.q[0])
        ])
        ee = self.forward_kinematics()
        return base, elbow, ee

    def compute_jacobian(self) -> np.ndarray:
        """Compute the Jacobian matrix."""
        return np.array([
            [-self.l1*np.sin(self.q[0]) - self.l2*np.sin(self.q[0]+self.q[1]),
             -self.l2*np.sin(self.q[0]+self.q[1])],
            [self.l1*np.cos(self.q[0]) + self.l2*np.cos(self.q[0]+self.q[1]),
             self.l2*np.cos(self.q[0]+self.q[1])]
        ])

    def step(self, dt: float = 0.02):
        """Perform one control step."""
        ee = self.forward_kinematics()
        error = self.target - ee

        # Stop if close enough
        if np.linalg.norm(error) < 0.01:
            self.last_dq_magnitude = 0.0
            return

        J = self.compute_jacobian()

        # Choose method based on toggle
        if self.use_jparse:
            J_inv = self.solver.compute(J)
        else:
            J_inv = self.solver.pinv(J)

        # Compute joint velocities
        v_des = self.k * error
        dq = np.asarray(J_inv @ v_des).flatten()

        # Track raw magnitude before limiting (shows singularity behavior)
        self.last_dq_magnitude = np.linalg.norm(dq)

        # Limit joint velocities
        max_dq = 2.0
        if np.max(np.abs(dq)) > max_dq:
            dq = dq * max_dq / np.max(np.abs(dq))

        # Update joints
        self.q += dq * dt

    def get_manipulability_ellipse(self, scale: float = 0.3) -> tuple:
        """
        Compute manipulability ellipse parameters.
        Returns (width, height, angle_deg) for the ellipse at end-effector.
        """
        J = self.compute_jacobian()
        # Manipulability matrix
        M = J @ J.T
        eigenvalues, eigenvectors = np.linalg.eig(M)

        # Sort by eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Ellipse parameters
        width = scale * np.sqrt(max(eigenvalues[0], 1e-6))
        height = scale * np.sqrt(max(eigenvalues[1], 1e-6))
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        return width, height, angle


class ArmVisualization:
    """Interactive matplotlib visualization."""

    def __init__(self, arm: PlanarArm2D):
        self.arm = arm

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_title('2D Arm Control with J-PARSE\n(Click to set target)')

        # Draw workspace circle
        workspace = Circle((0, 0), arm.l1 + arm.l2, fill=False,
                          linestyle='--', color='gray', alpha=0.5, label='Workspace')
        self.ax.add_patch(workspace)

        # Arm links
        self.link1, = self.ax.plot([], [], 'b-', linewidth=6, solid_capstyle='round')
        self.link2, = self.ax.plot([], [], 'c-', linewidth=5, solid_capstyle='round')

        # Joints
        self.joints, = self.ax.plot([], [], 'ko', markersize=10, zorder=5)

        # End-effector
        self.ee_marker, = self.ax.plot([], [], 'go', markersize=12, label='End-effector')

        # Target
        self.target_marker, = self.ax.plot([], [], 'rx', markersize=15,
                                           markeredgewidth=3, label='Target')

        # Manipulability ellipse
        self.ellipse = Ellipse((0, 0), 0.1, 0.1, angle=0, fill=False,
                               color='purple', linewidth=2, label='Manipulability')
        self.ax.add_patch(self.ellipse)

        # Trajectory trace
        self.trajectory_x = []
        self.trajectory_y = []
        self.trajectory_line, = self.ax.plot([], [], 'g-', alpha=0.3, linewidth=1)

        # Info text
        self.info_text = self.ax.text(-2.4, 2.2, '', fontsize=10,
                                      verticalalignment='top', fontfamily='monospace')

        self.ax.legend(loc='upper right')

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Connect keyboard event for toggle
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        """Handle keyboard input to toggle between J-PARSE and pinv."""
        if event.key in ['j', 'J', ' ']:
            self.arm.use_jparse = not self.arm.use_jparse
            method = "J-PARSE" if self.arm.use_jparse else "Pseudo-inverse"
            print(f"Switched to: {method}")
            # Clear trajectory when switching
            self.trajectory_x = []
            self.trajectory_y = []

    def on_click(self, event):
        """Handle mouse click to set new target."""
        if event.inaxes != self.ax:
            return

        # Set new target
        self.arm.target = np.array([event.xdata, event.ydata])

        # Clear trajectory
        self.trajectory_x = []
        self.trajectory_y = []

    def update(self, frame):
        """Animation update function."""
        # Step the arm
        self.arm.step(dt=0.02)

        # Get positions
        base, elbow, ee = self.arm.get_joint_positions()

        # Update arm visualization
        self.link1.set_data([base[0], elbow[0]], [base[1], elbow[1]])
        self.link2.set_data([elbow[0], ee[0]], [elbow[1], ee[1]])
        self.joints.set_data([base[0], elbow[0], ee[0]], [base[1], elbow[1], ee[1]])
        self.ee_marker.set_data([ee[0]], [ee[1]])
        self.target_marker.set_data([self.arm.target[0]], [self.arm.target[1]])

        # Update ellipse
        w, h, angle = self.arm.get_manipulability_ellipse()
        self.ellipse.set_center(ee)
        self.ellipse.width = 2 * w
        self.ellipse.height = 2 * h
        self.ellipse.angle = angle

        # Update trajectory
        self.trajectory_x.append(ee[0])
        self.trajectory_y.append(ee[1])
        if len(self.trajectory_x) > 500:  # Limit trace length
            self.trajectory_x = self.trajectory_x[-500:]
            self.trajectory_y = self.trajectory_y[-500:]
        self.trajectory_line.set_data(self.trajectory_x, self.trajectory_y)

        # Update info text
        J = self.arm.compute_jacobian()
        m = jparse.manipulability_measure(J)
        icn = jparse.inverse_condition_number(J)
        error = np.linalg.norm(self.arm.target - ee)
        method = "J-PARSE" if self.arm.use_jparse else "PINV"

        info = (
            f"Method: {method} (press 'j' to toggle)\n"
            f"Joint angles: q1={np.degrees(self.arm.q[0]):.1f}deg, "
            f"q2={np.degrees(self.arm.q[1]):.1f}deg\n"
            f"End-effector: ({ee[0]:.3f}, {ee[1]:.3f})\n"
            f"Target: ({self.arm.target[0]:.3f}, {self.arm.target[1]:.3f})\n"
            f"Position error: {error:.4f} m\n"
            f"Joint vel magnitude: {self.arm.last_dq_magnitude:.2f} rad/s\n"
            f"Manipulability: {m:.4f}\n"
            f"Inv. condition #: {icn:.4f}"
        )
        self.info_text.set_text(info)

        # Update title to show current method
        self.ax.set_title(f'2D Arm Control with {method}\n(Click to set target, press "j" to toggle method)')

        return (self.link1, self.link2, self.joints, self.ee_marker,
                self.target_marker, self.ellipse, self.trajectory_line, self.info_text)

    def run(self):
        """Start the animation."""
        self.anim = FuncAnimation(self.fig, self.update, frames=None,
                                  interval=20, blit=True, cache_frame_data=False)
        plt.show()


def main():
    print("=" * 60)
    print("2D Arm Visualization with J-PARSE Control")
    print("=" * 60)
    print("Controls:")
    print("  - Click anywhere to set a target position")
    print("  - Press 'j' or SPACE to toggle between J-PARSE and pinv")
    print("  - Purple ellipse shows manipulability (flattens near singularity)")
    print()
    print("To see J-PARSE advantage:")
    print("  1. Click near workspace boundary (where arm is stretched)")
    print("  2. Watch 'Joint vel magnitude' - pinv explodes near singularities!")
    print("  3. Toggle to J-PARSE to see bounded velocities")
    print("=" * 60)

    # Create arm and visualization
    arm = PlanarArm2D(l1=1.0, l2=1.0, gamma=0.1)
    viz = ArmVisualization(arm)
    viz.run()


if __name__ == "__main__":
    main()
