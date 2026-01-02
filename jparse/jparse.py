"""
J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities Effectively
"""

import numpy as np


class JParse:
    """
    J-PARSE implementation for singularity-robust inverse kinematics control.
    """

    def svd_compose(self, U, S, Vt):
        """
        Recomposes a matrix from its SVD components U, S, Vt.

        Parameters:
            U: Left singular vectors
            S: Singular values (1D array)
            Vt: Right singular vectors (transposed)

        Returns:
            Reconstructed matrix J_new
        """
        Sfull = np.zeros((U.shape[1], Vt.shape[0]))
        for row in range(Sfull.shape[0]):
            for col in range(Sfull.shape[1]):
                if row == col and row < len(S):
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
        Compute the J-PARSE inverse of a Jacobian matrix.

        Parameters:
            J: Jacobian matrix (m x n) numpy array
            jac_nullspace_bool: If True, also return the nullspace projection matrix
            gamma: Threshold gain for singular directions (default=0.1)
            singular_direction_gain_position: Gain for singular directions in position
            singular_direction_gain_orientation: Gain for singular directions in orientation
            position_dimensions: Number of dimensions for position (if None, uses J rows)
            angular_dimensions: Number of dimensions for orientation

        Returns:
            J_parse: The J-PARSE inverse (n x m) numpy matrix
            J_safety_nullspace (optional): Nullspace projection matrix (n x n)
        """
        # Perform SVD decomposition
        U, S, Vt = np.linalg.svd(J)

        # Find the adjusted condition number
        sigma_max = np.max(S)
        adjusted_condition_numbers = [sig / sigma_max for sig in S]

        # Find the projection Jacobian
        U_new_proj = []
        S_new_proj = []
        for col in range(len(S)):
            if S[col] > gamma * sigma_max:
                U_new_proj.append(np.matrix(U[:, col]).T)
                S_new_proj.append(S[col])
        U_new_proj = np.concatenate(U_new_proj, axis=1)
        J_proj = self.svd_compose(U_new_proj, S_new_proj, Vt)

        # Find the safety Jacobian
        S_new_safety = [
            s if (s / sigma_max) > gamma else gamma * sigma_max for s in S
        ]
        J_safety = self.svd_compose(U, S_new_safety, Vt)

        # Find the singular direction projection components
        U_new_sing = []
        Phi = []
        set_empty_bool = True
        for col in range(len(S)):
            if adjusted_condition_numbers[col] <= gamma:
                set_empty_bool = False
                U_new_sing.append(np.matrix(U[:, col]).T)
                Phi.append(adjusted_condition_numbers[col] / gamma)

        # Initialize singular projection matrix
        Phi_singular = np.zeros(U.shape)

        if not set_empty_bool:
            U_new_sing = np.matrix(np.concatenate(U_new_sing, axis=1))
            Phi_mat = np.matrix(np.diag(Phi))

            # Handle gain conditions
            if position_dimensions is None and angular_dimensions is None:
                gain_dimension = J.shape[0]
                gains = np.array(
                    [singular_direction_gain_position] * gain_dimension, dtype=float
                )
            elif angular_dimensions is None and position_dimensions is not None:
                gain_dimension = position_dimensions
                gains = np.array(
                    [singular_direction_gain_position] * gain_dimension, dtype=float
                )
            elif position_dimensions is None and angular_dimensions is not None:
                gain_dimension = angular_dimensions
                gains = np.array(
                    [singular_direction_gain_orientation] * gain_dimension, dtype=float
                )
            else:
                gains = np.array(
                    [singular_direction_gain_position] * position_dimensions
                    + [singular_direction_gain_orientation] * angular_dimensions,
                    dtype=float,
                )

            Kp_singular = np.diag(gains)
            Phi_singular = U_new_sing @ Phi_mat @ U_new_sing.T @ Kp_singular

        # Obtain pseudo-inverses
        J_safety_pinv = np.linalg.pinv(J_safety)
        J_proj_pinv = np.linalg.pinv(J_proj)

        if not set_empty_bool:
            J_parse = (
                J_safety_pinv @ J_proj @ J_proj_pinv + J_safety_pinv @ Phi_singular
            )
        else:
            J_parse = J_safety_pinv @ J_proj @ J_proj_pinv

        if jac_nullspace_bool:
            J_safety_nullspace = np.eye(J_safety.shape[1]) - J_safety_pinv @ J_safety
            return J_parse, J_safety_nullspace

        return J_parse
