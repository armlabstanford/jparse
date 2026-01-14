"""
Regression tests to ensure the new jparse_robotics package matches
the original JParseClass implementation from the notebook/ROS code.

These tests copy the original implementation and verify that the new
implementation produces identical results.
"""

import numpy as np
import pytest

import jparse_robotics as jparse


class OriginalJParseClass:
    """
    Original JParseClass implementation from jparse_collab_example.ipynb.

    This is copied verbatim from the notebook to serve as the reference
    implementation for regression testing.
    """

    def svd_compose(self, U, S, Vt):
        """
        This function takes SVD: U,S,V and recomposes them for J
        """
        Zero_concat = np.zeros((U.shape[0], Vt.shape[0] - len(S)))
        Sfull = np.zeros((U.shape[1], Vt.shape[0]))
        for row in range(Sfull.shape[0]):
            for col in range(Sfull.shape[1]):
                if row == col:
                    if row < len(S):
                        Sfull[row, col] = S[row]
        J_new = np.matrix(U) * Sfull * np.matrix(Vt)
        return J_new

    def JParse(
        self,
        J=None,
        jac_nullspace_bool=False,
        gamma=0.1,
        singular_direction_gain_position=1,
        singular_direction_gain_orientation=1,
        position_dimensions=None,
        angular_dimensions=None,
    ):
        """
        Original JParse implementation from the notebook.

        input: Jacobian J (m x n) numpy matrix
        Args:
          - jac_nullspace_bool (default=False): Set this to true of the nullspace of the jacobian is desired
          - gamma (default=0.1): threshold gain for singular directions
          - singular_direction_gain_position (default=1): gain for singular directions in position
          - singular_direction_gain_orientation (default=1): gain for singular directions in orientation
          - position_dimensions (default=None): the number of dimensions for the position
          - angular_dimensions (default=None): the number of dimensions for the orientation

        output:
          - J_parse (n x m) numpy matrix
          - (optional) J_safety_nullspace (n x n) numpy matrix
        """
        # Perform the SVD decomposition of the jacobian
        U, S, Vt = np.linalg.svd(J)
        # Find the adjusted condition number
        sigma_max = np.max(S)
        adjusted_condition_numbers = [sig / sigma_max for sig in S]

        # Find the projection Jacobian
        U_new_proj = []
        S_new_proj = []
        for col in range(len(S)):
            if S[col] > gamma * sigma_max:
                # Singular row
                U_new_proj.append(np.matrix(U[:, col]).T)
                S_new_proj.append(S[col])
        U_new_proj = np.concatenate(
            U_new_proj, axis=1
        )  # Careful which numpy version is being used for this!!!!!
        J_proj = self.svd_compose(U_new_proj, S_new_proj, Vt)

        # Find the safety jacboian
        S_new_safety = [
            s if (s / sigma_max) > gamma else gamma * sigma_max for s in S
        ]
        J_safety = self.svd_compose(U, S_new_safety, Vt)

        # Find the singular direction projection components
        U_new_sing = []
        Phi = []  # these will be the ratio of s_i/s_max
        set_empty_bool = True
        for col in range(len(S)):
            if adjusted_condition_numbers[col] <= gamma:
                set_empty_bool = False
                U_new_sing.append(np.matrix(U[:, col]).T)
                Phi.append(
                    adjusted_condition_numbers[col] / gamma
                )  # division by gamma for s/(s_max * gamma), gives smooth transition for Kp =1.0;

        # set an empty Phi_singular matrix, populate if there were any adjusted
        # condition numbers below the threshold
        Phi_singular = np.zeros(U.shape)  # initialize the singular projection matrix

        if set_empty_bool == False:
            # construct the new U, as there were singular directions
            U_new_sing = np.matrix(
                np.concatenate(U_new_sing, axis=1)
            )  # Careful which numpy version is being used for this!!!!!
            Phi_mat = np.matrix(np.diag(Phi))

            # Now handle the gain conditions
            if position_dimensions == None and angular_dimensions == None:
                # neither dimensions have been set, this is the default case
                gain_dimension = J.shape[0]
                gains = np.array(
                    [singular_direction_gain_position] * gain_dimension, dtype=float
                )
            elif angular_dimensions == None and position_dimensions != None:
                # only position dimensions have been set
                gain_dimension = position_dimensions
                gains = np.array(
                    [singular_direction_gain_position] * gain_dimension, dtype=float
                )
            elif position_dimensions == None and angular_dimensions != None:
                # only angular dimensions have been set
                gain_dimension = angular_dimensions
                gains = np.array(
                    [singular_direction_gain_orientation] * gain_dimension, dtype=float
                )
            else:
                # both position and angular dimensions are filled
                gains = np.array(
                    [singular_direction_gain_position] * position_dimensions
                    + [singular_direction_gain_orientation] * angular_dimensions,
                    dtype=float,
                )
            # now put them into a matrix:
            Kp_singular = np.diag(gains)

            # Now put it all together:
            Phi_singular = U_new_sing @ Phi_mat @ U_new_sing.T @ Kp_singular

        # Obtain psuedo-inverse of the safety jacobian and the projection jacobian
        J_safety_pinv = np.linalg.pinv(J_safety)
        J_proj_pinv = np.linalg.pinv(J_proj)

        if set_empty_bool == False:
            J_parse = (
                J_safety_pinv @ J_proj @ J_proj_pinv + J_safety_pinv @ Phi_singular
            )
        else:
            J_parse = J_safety_pinv @ J_proj @ J_proj_pinv

        if jac_nullspace_bool == True:
            # Find the nullspace of the jacobian
            J_safety_nullspace = np.eye(J_safety.shape[1]) - J_safety_pinv @ J_safety
            return J_parse, J_safety_nullspace

        return J_parse


class TestRegressionAgainstOriginal:
    """
    Test that new implementation matches original implementation exactly.
    """

    @pytest.fixture
    def original_jparse(self):
        """Create original JParseClass instance."""
        return OriginalJParseClass()

    @pytest.fixture
    def new_jparse(self):
        """Create new JParseCore instance."""
        return jparse.JParseCore(gamma=0.1)

    def test_2dof_well_conditioned(self, original_jparse, new_jparse):
        """Test 2-DOF planar arm in well-conditioned configuration."""
        # 2-link planar arm at theta1=pi/4, theta2=pi/4
        l1, l2 = 1.0, 1.0
        theta1, theta2 = np.pi / 4, np.pi / 4

        J = np.array(
            [
                [
                    -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2),
                    -l2 * np.sin(theta1 + theta2),
                ],
                [
                    l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
                    l2 * np.cos(theta1 + theta2),
                ],
            ]
        )

        J_parse_original = np.asarray(original_jparse.JParse(J=J, gamma=0.1))
        J_parse_new = np.asarray(new_jparse.compute(J))

        assert np.allclose(J_parse_original, J_parse_new, atol=1e-10)

    def test_2dof_near_singularity(self, original_jparse, new_jparse):
        """Test 2-DOF planar arm near singularity."""
        # Near singular configuration
        l1, l2 = 1.0, 1.0
        theta1, theta2 = 0.1, 0.1

        J = np.array(
            [
                [
                    -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2),
                    -l2 * np.sin(theta1 + theta2),
                ],
                [
                    l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
                    l2 * np.cos(theta1 + theta2),
                ],
            ]
        )

        J_parse_original = np.asarray(original_jparse.JParse(J=J, gamma=0.1))
        J_parse_new = np.asarray(new_jparse.compute(J))

        assert np.allclose(J_parse_original, J_parse_new, atol=1e-10)

    def test_3dof_redundant(self, original_jparse, new_jparse):
        """Test 3-DOF planar arm (redundant system)."""
        # 3-link planar arm
        l1, l2, l3 = 1.0, 1.0, 1.0
        q1, q2, q3 = np.pi / 4, np.pi / 4, np.pi / 4

        J = np.array(
            [
                [
                    -l1 * np.sin(q1)
                    - l2 * np.sin(q1 + q2)
                    - l3 * np.sin(q1 + q2 + q3),
                    -l2 * np.sin(q1 + q2) - l3 * np.sin(q1 + q2 + q3),
                    -l3 * np.sin(q1 + q2 + q3),
                ],
                [
                    l1 * np.cos(q1)
                    + l2 * np.cos(q1 + q2)
                    + l3 * np.cos(q1 + q2 + q3),
                    l2 * np.cos(q1 + q2) + l3 * np.cos(q1 + q2 + q3),
                    l3 * np.cos(q1 + q2 + q3),
                ],
            ]
        )

        J_parse_original = np.asarray(original_jparse.JParse(J=J, gamma=0.1))
        J_parse_new = np.asarray(new_jparse.compute(J))

        assert np.allclose(J_parse_original, J_parse_new, atol=1e-10)

    def test_with_nullspace(self, original_jparse, new_jparse):
        """Test nullspace computation matches."""
        J = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

        J_parse_orig, N_orig = original_jparse.JParse(
            J=J, gamma=0.1, jac_nullspace_bool=True
        )
        J_parse_new, N_new = new_jparse.compute(J, return_nullspace=True)

        assert np.allclose(np.asarray(J_parse_orig), np.asarray(J_parse_new), atol=1e-10)
        assert np.allclose(np.asarray(N_orig), np.asarray(N_new), atol=1e-10)

    def test_with_position_gains(self, original_jparse, new_jparse):
        """Test with position gains specified."""
        J = np.array([[1.0, 0.5], [0.5, 0.1]])  # Near singular

        J_parse_orig = original_jparse.JParse(
            J=J,
            gamma=0.1,
            singular_direction_gain_position=2.0,
            position_dimensions=2,
        )
        J_parse_new = new_jparse.compute(
            J,
            singular_direction_gain_position=2.0,
            position_dimensions=2,
        )

        assert np.allclose(np.asarray(J_parse_orig), np.asarray(J_parse_new), atol=1e-10)

    def test_6dof_full_pose(self, original_jparse):
        """Test with 6x7 Jacobian (full pose control, redundant)."""
        np.random.seed(42)  # For reproducibility
        J = np.random.randn(6, 7)

        # Create new instance with same gamma
        new_jparse = jparse.JParseCore(gamma=0.1)

        J_parse_orig = original_jparse.JParse(
            J=J,
            gamma=0.1,
            singular_direction_gain_position=1.5,
            singular_direction_gain_orientation=2.0,
            position_dimensions=3,
            angular_dimensions=3,
        )
        J_parse_new = new_jparse.compute(
            J,
            singular_direction_gain_position=1.5,
            singular_direction_gain_angular=2.0,
            position_dimensions=3,
            angular_dimensions=3,
        )

        assert np.allclose(np.asarray(J_parse_orig), np.asarray(J_parse_new), atol=1e-10)

    def test_different_gamma_values(self, original_jparse):
        """Test with different gamma values."""
        J = np.array([[1.0, 0.1], [0.1, 1.0]])

        for gamma in [0.05, 0.1, 0.2, 0.3, 0.5]:
            new_jparse = jparse.JParseCore(gamma=gamma)

            J_parse_orig = original_jparse.JParse(J=J, gamma=gamma)
            J_parse_new = new_jparse.compute(J)

            assert np.allclose(
                np.asarray(J_parse_orig), np.asarray(J_parse_new), atol=1e-10
            ), f"Failed for gamma={gamma}"

    def test_singular_jacobian(self, original_jparse, new_jparse):
        """Test with exactly singular Jacobian."""
        J = np.array([[1.0, 1.0], [1.0, 1.0]])  # Rank 1

        J_parse_orig = original_jparse.JParse(J=J, gamma=0.1)
        J_parse_new = new_jparse.compute(J)

        assert np.allclose(np.asarray(J_parse_orig), np.asarray(J_parse_new), atol=1e-10)

    def test_identity_jacobian(self, original_jparse, new_jparse):
        """Test with identity Jacobian (well-conditioned)."""
        J = np.eye(3)

        J_parse_orig = original_jparse.JParse(J=J, gamma=0.1)
        J_parse_new = new_jparse.compute(J)

        assert np.allclose(np.asarray(J_parse_orig), np.asarray(J_parse_new), atol=1e-10)

    def test_random_jacobians(self, original_jparse):
        """Test with many random Jacobians for robustness."""
        np.random.seed(123)

        shapes = [(2, 2), (2, 3), (3, 3), (3, 5), (6, 6), (6, 7)]

        for shape in shapes:
            for _ in range(5):  # 5 random tests per shape
                J = np.random.randn(*shape)
                new_jparse = jparse.JParseCore(gamma=0.1)

                J_parse_orig = original_jparse.JParse(J=J, gamma=0.1)
                J_parse_new = new_jparse.compute(J)

                assert np.allclose(
                    np.asarray(J_parse_orig), np.asarray(J_parse_new), atol=1e-10
                ), f"Failed for shape {shape}"


class TestVelocityControlEquivalence:
    """
    Test that velocity control produces same results with both implementations.
    """

    def test_velocity_control_trajectory(self):
        """
        Simulate a trajectory and verify both implementations
        produce the same joint velocities.
        """
        original = OriginalJParseClass()
        new = jparse.JParseCore(gamma=0.1)

        # 2-link planar arm
        l1, l2 = 1.0, 1.0
        theta1, theta2 = np.pi / 4, np.pi / 4

        # Control parameters
        dt = 0.01
        k = 2.0

        # Track positions
        pos_original = []
        pos_new = []

        theta1_orig, theta2_orig = theta1, theta2
        theta1_new, theta2_new = theta1, theta2

        for i in range(100):
            t = i * dt

            # Desired position (circular trajectory)
            x_des = 1.0 + 0.5 * np.cos(0.5 * t)
            y_des = 0.5 * np.sin(0.5 * t)

            # Original implementation
            x_orig = l1 * np.cos(theta1_orig) + l2 * np.cos(theta1_orig + theta2_orig)
            y_orig = l1 * np.sin(theta1_orig) + l2 * np.sin(theta1_orig + theta2_orig)

            J_orig = np.array(
                [
                    [
                        -l1 * np.sin(theta1_orig) - l2 * np.sin(theta1_orig + theta2_orig),
                        -l2 * np.sin(theta1_orig + theta2_orig),
                    ],
                    [
                        l1 * np.cos(theta1_orig) + l2 * np.cos(theta1_orig + theta2_orig),
                        l2 * np.cos(theta1_orig + theta2_orig),
                    ],
                ]
            )

            error_orig = np.array([x_des - x_orig, y_des - y_orig])
            J_parse_orig = original.JParse(J=J_orig, gamma=0.1)
            dq_orig = np.asarray(J_parse_orig @ (k * error_orig)).flatten()

            theta1_orig += dq_orig[0] * dt
            theta2_orig += dq_orig[1] * dt
            pos_original.append([x_orig, y_orig])

            # New implementation
            x_new = l1 * np.cos(theta1_new) + l2 * np.cos(theta1_new + theta2_new)
            y_new = l1 * np.sin(theta1_new) + l2 * np.sin(theta1_new + theta2_new)

            J_new = np.array(
                [
                    [
                        -l1 * np.sin(theta1_new) - l2 * np.sin(theta1_new + theta2_new),
                        -l2 * np.sin(theta1_new + theta2_new),
                    ],
                    [
                        l1 * np.cos(theta1_new) + l2 * np.cos(theta1_new + theta2_new),
                        l2 * np.cos(theta1_new + theta2_new),
                    ],
                ]
            )

            error_new = np.array([x_des - x_new, y_des - y_new])
            J_parse_new = new.compute(J_new)
            dq_new = np.asarray(J_parse_new @ (k * error_new)).flatten()

            theta1_new += dq_new[0] * dt
            theta2_new += dq_new[1] * dt
            pos_new.append([x_new, y_new])

        # Verify trajectories match
        pos_original = np.array(pos_original)
        pos_new = np.array(pos_new)

        assert np.allclose(pos_original, pos_new, atol=1e-10)
        assert np.isclose(theta1_orig, theta1_new, atol=1e-10)
        assert np.isclose(theta2_orig, theta2_new, atol=1e-10)
