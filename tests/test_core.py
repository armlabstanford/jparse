"""Tests for the core J-PARSE algorithm."""

import numpy as np
import pytest

from jparse_robotics import JParseCore, inverse_condition_number, manipulability_measure


class TestJParseCore:
    """Test the JParseCore class."""

    def test_init_default_gamma(self):
        """Test default gamma value."""
        jparse = JParseCore()
        assert jparse.gamma == 0.1

    def test_init_custom_gamma(self):
        """Test custom gamma value."""
        jparse = JParseCore(gamma=0.2)
        assert jparse.gamma == 0.2

    def test_init_invalid_gamma(self):
        """Test that invalid gamma raises error."""
        with pytest.raises(ValueError):
            JParseCore(gamma=0)
        with pytest.raises(ValueError):
            JParseCore(gamma=1)
        with pytest.raises(ValueError):
            JParseCore(gamma=-0.1)
        with pytest.raises(ValueError):
            JParseCore(gamma=1.5)

    def test_2dof_planar_well_conditioned(self):
        """Test with well-conditioned 2-DOF planar Jacobian."""
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

        jparse = JParseCore(gamma=0.1)
        J_parse = jparse.compute(J)

        # Check shape
        assert J_parse.shape == (2, 2)

        # Check it's a valid pseudo-inverse (J @ J_parse @ J ≈ J)
        assert np.allclose(J @ J_parse @ J, J, atol=1e-6)

        # For well-conditioned case, should be close to pinv
        J_pinv = np.linalg.pinv(J)
        assert np.allclose(J_parse, J_pinv, atol=0.1)

    def test_near_singularity_no_nan(self):
        """Test that near-singular Jacobian doesn't produce NaN."""
        # Nearly singular Jacobian (rows are almost parallel)
        J = np.array([[1.0, 1.0], [1.0, 1.001]])

        jparse = JParseCore(gamma=0.1)
        J_parse = jparse.compute(J)

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(J_parse))

    def test_singular_jacobian(self):
        """Test with exactly singular Jacobian."""
        # Rank-deficient Jacobian
        J = np.array([[1.0, 1.0], [1.0, 1.0]])

        jparse = JParseCore(gamma=0.1)
        J_parse = jparse.compute(J)

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(J_parse))

    def test_redundant_system(self):
        """Test with redundant system (more joints than task space dims)."""
        # 2x3 Jacobian (redundant)
        J = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

        jparse = JParseCore(gamma=0.1)
        J_parse = jparse.compute(J)

        # Check shape
        assert J_parse.shape == (3, 2)

        # Check it's a valid pseudo-inverse
        assert np.allclose(J @ J_parse @ J, J, atol=1e-6)

    def test_nullspace_computation(self):
        """Test nullspace computation for redundant system."""
        # 2x3 Jacobian (redundant)
        J = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

        jparse = JParseCore(gamma=0.1)
        J_parse, N = jparse.compute(J, return_nullspace=True)

        # Check shapes
        assert J_parse.shape == (3, 2)
        assert N.shape == (3, 3)

        # Nullspace should project to approximately zero in task space
        null_vec = N @ np.array([1.0, 1.0, 1.0])
        task_effect = J @ null_vec
        assert np.allclose(task_effect, 0, atol=0.1)

    def test_position_only_gains(self):
        """Test with position-only dimension specification."""
        J = np.array([[1.0, 0.0], [0.0, 1.0]])

        jparse = JParseCore(gamma=0.1)
        J_parse = jparse.compute(
            J,
            singular_direction_gain_position=2.0,
            position_dimensions=2,
        )

        assert J_parse.shape == (2, 2)
        assert np.all(np.isfinite(J_parse))

    def test_full_pose_gains(self):
        """Test with both position and angular gains."""
        # 6x7 Jacobian (typical 7-DOF arm)
        J = np.random.randn(6, 7)

        jparse = JParseCore(gamma=0.1)
        J_parse = jparse.compute(
            J,
            singular_direction_gain_position=1.5,
            singular_direction_gain_angular=2.0,
            position_dimensions=3,
            angular_dimensions=3,
        )

        assert J_parse.shape == (7, 6)
        assert np.all(np.isfinite(J_parse))

    def test_pinv_method(self):
        """Test the pinv comparison method."""
        J = np.array([[1.0, 0.5], [0.5, 1.0]])

        jparse = JParseCore(gamma=0.1)
        J_pinv = jparse.pinv(J)

        # Should match numpy's pinv
        assert np.allclose(J_pinv, np.linalg.pinv(J))

    def test_damped_least_squares(self):
        """Test the damped least squares comparison method."""
        J = np.array([[1.0, 0.0], [0.0, 1.0]])
        damping = 0.1

        jparse = JParseCore(gamma=0.1)
        J_dls = jparse.damped_least_squares(J, damping=damping)

        # Compute expected DLS
        expected = np.linalg.inv(J.T @ J + damping**2 * np.eye(2)) @ J.T

        assert np.allclose(J_dls, expected)

    def test_damped_least_squares_with_nullspace(self):
        """Test DLS with nullspace output."""
        J = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

        jparse = JParseCore(gamma=0.1)
        J_dls, N = jparse.damped_least_squares(J, damping=0.01, return_nullspace=True)

        assert J_dls.shape == (3, 2)
        assert N.shape == (3, 3)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_manipulability_measure(self):
        """Test manipulability measure computation."""
        # Identity-like Jacobian should have high manipulability
        J = np.array([[1.0, 0.0], [0.0, 1.0]])
        m = manipulability_measure(J)
        assert m > 0
        assert np.isclose(m, 1.0)

        # Near-singular should have low manipulability
        J_singular = np.array([[1.0, 1.0], [1.0, 1.001]])
        m_singular = manipulability_measure(J_singular)
        assert m_singular < m

    def test_inverse_condition_number(self):
        """Test inverse condition number computation."""
        # Well-conditioned
        J = np.array([[1.0, 0.0], [0.0, 1.0]])
        icn = inverse_condition_number(J)
        assert np.isclose(icn, 1.0)

        # Less well-conditioned
        J2 = np.array([[1.0, 0.0], [0.0, 0.1]])
        icn2 = inverse_condition_number(J2)
        assert icn2 < icn
        assert np.isclose(icn2, 0.1)

    def test_inverse_condition_number_range(self):
        """Test that inverse condition number is in [0, 1]."""
        for _ in range(10):
            J = np.random.randn(3, 4)
            icn = inverse_condition_number(J)
            assert 0 <= icn <= 1


class TestEdgeCases:
    """Test edge cases and special inputs."""

    def test_1x1_jacobian(self):
        """Test with 1x1 Jacobian."""
        J = np.array([[2.0]])

        jparse = JParseCore(gamma=0.1)
        J_parse = jparse.compute(J)

        assert J_parse.shape == (1, 1)
        assert np.isclose(J_parse[0, 0], 0.5)

    def test_tall_jacobian(self):
        """Test with tall Jacobian (more outputs than inputs)."""
        J = np.array([[1.0], [0.5], [0.25]])

        jparse = JParseCore(gamma=0.1)
        J_parse = jparse.compute(J)

        assert J_parse.shape == (1, 3)

    def test_very_small_gamma(self):
        """Test with very small gamma."""
        J = np.array([[1.0, 0.5], [0.5, 1.0]])

        jparse = JParseCore(gamma=0.01)
        J_parse = jparse.compute(J)

        # Should be close to standard pinv for well-conditioned case
        assert np.allclose(J_parse, np.linalg.pinv(J), atol=0.01)

    def test_large_gamma(self):
        """Test with gamma close to 1."""
        J = np.array([[1.0, 0.5], [0.5, 1.0]])

        jparse = JParseCore(gamma=0.99)
        J_parse = jparse.compute(J)

        # Should still produce valid output
        assert np.all(np.isfinite(J_parse))
