"""Tests for the Robot class with Pinocchio backend."""

import numpy as np
import pytest

# Skip all tests in this module if Pinocchio is not available
pinocchio = pytest.importorskip("pinocchio")


class TestRobotBasics:
    """Basic tests for Robot class."""

    @pytest.fixture
    def simple_urdf(self, tmp_path):
        """Create a simple 2-DOF planar arm URDF for testing."""
        urdf_content = """<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.5 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <link name="link2">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.5 0 0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <link name="ee_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1.0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>

  <joint name="ee_joint" type="fixed">
    <parent link="link2"/>
    <child link="ee_link"/>
    <origin xyz="1.0 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""
        urdf_path = tmp_path / "simple_arm.urdf"
        urdf_path.write_text(urdf_content)
        return str(urdf_path)

    def test_from_urdf_file(self, simple_urdf):
        """Test loading robot from URDF file."""
        from jparse_robotics import Robot

        robot = Robot.from_urdf(simple_urdf, "base_link", "ee_link")

        assert robot.num_joints == 2
        assert robot.gamma == 0.1

    def test_from_urdf_string(self, simple_urdf):
        """Test loading robot from URDF string."""
        from jparse_robotics import Robot

        with open(simple_urdf, "r") as f:
            urdf_string = f.read()

        robot = Robot.from_urdf(urdf_string, "base_link", "ee_link")
        assert robot.num_joints == 2

    def test_from_urdf_custom_gamma(self, simple_urdf):
        """Test loading robot with custom gamma."""
        from jparse_robotics import Robot

        robot = Robot.from_urdf(simple_urdf, "base_link", "ee_link", gamma=0.2)
        assert robot.gamma == 0.2

    def test_from_urdf_invalid_end_link(self, simple_urdf):
        """Test that invalid end link raises error."""
        from jparse_robotics import Robot

        with pytest.raises(ValueError, match="not found"):
            Robot.from_urdf(simple_urdf, "base_link", "nonexistent_link")

    def test_from_urdf_file_not_found(self):
        """Test that missing file raises error."""
        from jparse_robotics import Robot

        with pytest.raises(FileNotFoundError):
            Robot.from_urdf("/nonexistent/path/robot.urdf", "base", "ee")

    def test_gamma_setter(self, simple_urdf):
        """Test gamma property setter."""
        from jparse_robotics import Robot

        robot = Robot.from_urdf(simple_urdf, "base_link", "ee_link")
        robot.gamma = 0.15
        assert robot.gamma == 0.15

    def test_gamma_setter_invalid(self, simple_urdf):
        """Test gamma setter with invalid value."""
        from jparse_robotics import Robot

        robot = Robot.from_urdf(simple_urdf, "base_link", "ee_link")
        with pytest.raises(ValueError):
            robot.gamma = 0


class TestRobotKinematics:
    """Test Robot kinematics methods."""

    @pytest.fixture
    def robot(self, tmp_path):
        """Create a test robot."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="link1"><inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="link2"><inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  <link name="ee"><inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial></link>

  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="1.0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
  <joint name="ee_joint" type="fixed">
    <parent link="link2"/><child link="ee"/>
    <origin xyz="1.0 0 0"/>
  </joint>
</robot>"""
        urdf_path = tmp_path / "test_arm.urdf"
        urdf_path.write_text(urdf_content)

        from jparse_robotics import Robot

        return Robot.from_urdf(str(urdf_path), "base", "ee")

    def test_jacobian_shape(self, robot):
        """Test Jacobian has correct shape."""
        q = np.zeros(robot.num_joints)
        J = robot.jacobian(q)

        assert J.shape == (6, robot.num_velocities)

    def test_jacobian_finite(self, robot):
        """Test Jacobian is finite for various configurations."""
        for _ in range(10):
            q = np.random.uniform(-np.pi, np.pi, robot.num_joints)
            J = robot.jacobian(q)
            assert np.all(np.isfinite(J))

    def test_forward_kinematics(self, robot):
        """Test forward kinematics."""
        q = np.zeros(robot.num_joints)
        pos, rot = robot.forward_kinematics(q)

        assert pos.shape == (3,)
        assert rot.shape == (3, 3)

        # At zero config, end-effector should be at (2, 0, 0) for 2-link arm
        assert np.allclose(pos, [2.0, 0.0, 0.0], atol=1e-6)

        # Rotation should be identity
        assert np.allclose(rot, np.eye(3), atol=1e-6)

    def test_jparse_shape(self, robot):
        """Test J-PARSE output shape."""
        q = np.zeros(robot.num_joints)
        J_parse = robot.jparse(q)

        assert J_parse.shape == (robot.num_velocities, 6)

    def test_jparse_position_only(self, robot):
        """Test J-PARSE with position only."""
        q = np.zeros(robot.num_joints)
        J_parse = robot.jparse(q, position_only=True)

        assert J_parse.shape == (robot.num_velocities, 3)

    def test_jparse_with_nullspace(self, robot):
        """Test J-PARSE with nullspace output."""
        q = np.zeros(robot.num_joints)
        J_parse, N = robot.jparse(q, return_nullspace=True)

        assert J_parse.shape == (robot.num_velocities, 6)
        assert N.shape == (robot.num_velocities, robot.num_velocities)

    def test_jparse_finite(self, robot):
        """Test J-PARSE is finite for various configurations."""
        for _ in range(10):
            q = np.random.uniform(-np.pi, np.pi, robot.num_joints)
            J_parse = robot.jparse(q)
            assert np.all(np.isfinite(J_parse))

    def test_pinv(self, robot):
        """Test pseudo-inverse method."""
        q = np.zeros(robot.num_joints)
        J_pinv = robot.pinv(q)

        assert J_pinv.shape == (robot.num_velocities, 6)

    def test_damped_least_squares(self, robot):
        """Test damped least squares method."""
        q = np.zeros(robot.num_joints)
        J_dls = robot.damped_least_squares(q, damping=0.01)

        assert J_dls.shape == (robot.num_velocities, 6)

    def test_manipulability(self, robot):
        """Test manipulability computation."""
        q = np.zeros(robot.num_joints)
        m = robot.manipulability(q)

        assert m >= 0
        assert np.isfinite(m)

    def test_inverse_condition_number(self, robot):
        """Test inverse condition number computation."""
        q = np.zeros(robot.num_joints)
        icn = robot.inverse_condition_number(q)

        assert 0 <= icn <= 1


class TestRobotVelocityControl:
    """Test velocity control scenarios."""

    @pytest.fixture
    def robot(self, tmp_path):
        """Create a 7-DOF robot for testing."""
        # Create a simple chain
        links = ""
        joints = ""
        prev_link = "base"

        for i in range(7):
            links += f"""
  <link name="link{i}">
    <inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>"""
            axis = ["1 0 0", "0 1 0", "0 0 1"][i % 3]
            joints += f"""
  <joint name="j{i}" type="revolute">
    <parent link="{prev_link}"/><child link="link{i}"/>
    <origin xyz="0.3 0 0"/><axis xyz="{axis}"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>"""
            prev_link = f"link{i}"

        links += """
  <link name="ee">
    <inertial><mass value="0.1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
  </link>"""
        joints += f"""
  <joint name="ee_joint" type="fixed">
    <parent link="link6"/><child link="ee"/>
    <origin xyz="0.3 0 0"/>
  </joint>"""

        urdf_content = f"""<?xml version="1.0"?>
<robot name="arm7">
  <link name="base"><inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
  {links}
  {joints}
</robot>"""
        urdf_path = tmp_path / "arm7.urdf"
        urdf_path.write_text(urdf_content)

        from jparse_robotics import Robot

        return Robot.from_urdf(str(urdf_path), "base", "ee")

    def test_velocity_control_direction(self, robot):
        """Test that velocity control moves in desired direction."""
        q = np.zeros(robot.num_joints)

        # Get current position
        pos0, _ = robot.forward_kinematics(q)

        # Desired velocity in x direction
        v_des = np.array([0.1, 0, 0, 0, 0, 0])

        # Get joint velocities
        J_parse = robot.jparse(q)
        dq = J_parse @ v_des

        # Simulate one step
        dt = 0.01
        q_new = q + dq * dt

        # Get new position
        pos1, _ = robot.forward_kinematics(q_new)

        # Position should have moved in x direction
        delta = pos1 - pos0
        assert delta[0] > 0  # Moved in +x

    def test_nullspace_motion(self, robot):
        """Test that nullspace motion doesn't affect task."""
        q = np.zeros(robot.num_joints)

        # Get current position
        pos0, _ = robot.forward_kinematics(q)

        # Get J-PARSE with nullspace
        J_parse, N = robot.jparse(q, return_nullspace=True)

        # Create nullspace motion
        null_motion = N @ np.ones(robot.num_joints)

        # Apply nullspace motion
        dt = 0.01
        q_new = q + null_motion * dt

        # Get new position
        pos1, _ = robot.forward_kinematics(q_new)

        # Position change should be minimal
        delta = np.linalg.norm(pos1 - pos0)
        assert delta < 0.01  # Small movement
