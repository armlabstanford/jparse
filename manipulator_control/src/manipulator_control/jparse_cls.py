#!/usr/bin/env python3
import rospy
import numpy as np
#access pykdl for jacobian and other matricies
import PyKDL as kdl
from urdf_parser_py.urdf import Robot
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_kinematics import kdl_to_mat
from pykdl_utils.kdl_kinematics import joint_list_to_kdl
#for plotting ellipsoids
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Pose, Point, PoseStamped


class JParseClass(object):
    def __init__(self, base_link="base_link", end_link="end_effector_link"):
        # Initialize any necessary variables or parameters here
        """
        Base link: The base link of the robot.
        End link: The end link of the robot.
        """
        self.base_link = base_link
        self.end_link = end_link  
        self.urdf = Robot.from_parameter_server()
        self.kdl_tree = kdl_tree_from_urdf_model(self.urdf)
        self.chain = self.kdl_tree.getChain(base_link, end_link)
        self._fk_kdl = kdl.ChainFkSolverPos_recursive(self.chain)
        self._ik_v_kdl = kdl.ChainIkSolverVel_pinv(self.chain)
        self._ik_p_kdl = kdl.ChainIkSolverPos_NR(self.chain, self._fk_kdl, self._ik_v_kdl)
        self._jac_kdl = kdl.ChainJntToJacSolver(self.chain)
        self._dyn_kdl = kdl.ChainDynParam(self.chain, kdl.Vector(0,0,-9.81))
        self.kdl_kin = KDLKinematics(self.urdf, base_link, end_link)
        self.num_joints = self.kdl_kin.num_joints
        self.joint_names = self.kdl_kin.get_joint_names()  
        self.marker_pub = rospy.Publisher('/jparse_ellipsoid_marker', MarkerArray, queue_size=10)
        
        self.J_prev = None
        self.J_prev_time = None
        
    def jacobian(self, q=[]):
        """
        Computes the Jacobian matrix for the given joint configuration.

        Parameters:
        q (list): A list of joint positions.

        Returns:
        numpy.ndarray: The Jacobian matrix as a numpy array.
        
        This function converts the joint positions to a KDL joint array, computes the Jacobian using KDL,
        and then converts the resulting KDL Jacobian matrix to a numpy array.
        """
        j_kdl = kdl.Jacobian(self.num_joints)
        q_kdl = joint_list_to_kdl(q)
        self._jac_kdl.JntToJac(q_kdl, j_kdl)
        J_mat = kdl_to_mat(j_kdl) #converts kdl to numpy matrix
        return J_mat

    def jacobian_dot(self, q=[], position_only=False , approx=True):
        """
        Computes the time derivative of the Jacobian matrix for the given joint configuration and joint velocities.

        Parameters:
        q (list): A list of joint positions.

        Returns:
        numpy.ndarray: The time derivative of the Jacobian matrix as a numpy array.
        """
        J = self.jacobian(q)
        if position_only == True:
            #if we are only interested in the position, return the first 3 rows of the jacobian
            J= J[:3, :]

        if approx == True:
            #This is the approximate method for calculating the time derivative of the jacobian
            J_dot = np.zeros(J.shape)
            q_plus = q.copy()
            q_minus = q.copy()
            for i in range(self.num_joints):
                q_plus[i] += 0.0001
                q_minus[i] -= 0.0001
            J_plus = self.jacobian(q_plus)
            J_minus = self.jacobian(q_minus)
            J_dot = (J_plus - J_minus) / 0.0002
            return J_dot
        else:
            #This is the exact method for calculating the time derivative of the jacobian
            if self.J_prev is None:
                self.J_prev = J
                self.J_prev_time = rospy.Time.now()
                J_dot = np.zeros(J.shape)
            else:
                dt = (rospy.Time.now() - self.J_prev_time).to_sec()
                J_dot = (J - self.J_prev) / dt
                self.J_prev = J
            return J_dot

    def svd_compose(self,U,S,Vt):
        """
        This function takes SVD: U,S,V and recomposes them for J
        """
        Zero_concat = np.zeros((U.shape[0],Vt.shape[0]-len(S)))
        Sfull = np.zeros((U.shape[1],Vt.shape[0]))
        for row in range(Sfull.shape[0]):
            for col in range(Sfull.shape[1]):
              if row == col:
                  if row < len(S):        
                      Sfull[row,col] = S[row]
        J_new =np.matrix(U)*Sfull*np.matrix(Vt)
        return J_new

    def manipulability_measure(self, q=[], use_inv_cond_number=False):
        """
        This function computes the manipulability measure for the given joint configuration.
        """
        #if we are using the manipulability measure, return that
        J = self.jacobian(q)
        return np.sqrt(np.linalg.det(J @ J.T))

    def inverse_condition_number(self, q=[]):
        """
        This function computes the inverse of the condition number for the given joint configuration.
        """
        J = self.jacobian(q)
        U, S, Vt = np.linalg.svd(J)
        #find the min and max singular values
        sigma_min = np.min(S)
        sigma_max = np.max(S)
        inv_cond_number = sigma_min/sigma_max
        return inv_cond_number        

    def JacobianPseudoInverse(self, q=[], position_only=False, jac_nullspace_bool = False):
        """
        This function computes the pseudo-inverse of the Jacobian matrix for the given joint configuration.
        """
        J = self.jacobian(q)
        if position_only == True:
            #if we are only interested in the position, return the first 3 rows of the jacobian
            J= J[:3, :]
        J_pinv = np.linalg.pinv(J)
        if jac_nullspace_bool == True:
            #Find the nullspace of the jacobian
            J_nullspace = np.eye(J.shape[1]) - np.linalg.pinv(J) @ J
            return J_pinv, J_nullspace
        return J_pinv

    def JParse(self, q=[], gamma=0.1, position_only=False, jac_nullspace_bool = False , ks = 1, singular_direction_gain_position=1, singular_direction_gain_angular=1, verbose=False, publish_jparse_ellipses=False, internal_marker_flag=False, end_effector_pose=None, safety_only = False, safety_projection_only = False, ):
        """
        This function computes the JParse matrix for the given joint configuration and gamma value.
        This function should return the JParse matrix as a numpy array.
        - gamma is the threshold value for the adjusted condition number.
        - position_only is a boolean flag to indicate whether to use only the position part of the Jacobian.
        - singular_direction_gain is the gain for the singular direction projection matrix. Nominal values range from 1 to 10.
        """
        #obtain the original jacobian
        J = self.jacobian(q)

        if position_only == True:
            #if we are only interested in the position, return the first 3 rows of the jacobian
            J= J[:3, :]

        #Perform the SVD decomposition of the jacobian
        U, S, Vt = np.linalg.svd(J)
        #Find the adjusted condition number
        sigma_max = np.max(S)
        adjusted_condition_numbers = [sig / sigma_max for sig in S]

        if verbose == True:
            print("adjusted_condition_numbers:", adjusted_condition_numbers)
        #Find the projection Jacobian
        U_new_proj = []
        S_new_proj = []
        for col in range(len(S)):
            if S[col] > gamma*sigma_max:
                #Singular row
                U_new_proj.append(np.matrix(U[:,col]).T)
                S_new_proj.append(S[col])
        U_new_proj = np.concatenate(U_new_proj,axis=0).T
        J_proj = self.svd_compose(U_new_proj, S_new_proj, Vt)

        #Find the safety jacboian
        S_new_safety = [s if (s/sigma_max) > gamma else gamma*sigma_max for s in S]
        J_safety = self.svd_compose(U,S_new_safety,Vt)

        #Find the singular direction projection components
        U_new_sing = []
        Phi = [] #these will be the ratio of s_i/s_max
        set_empty_bool = True
        for col in range(len(S)):
            if adjusted_condition_numbers[col] <= gamma:
                set_empty_bool = False
                U_new_sing.append(np.matrix(U[:,col]).T)
                Phi.append(adjusted_condition_numbers[col]/gamma) #division by gamma for s/(s_max * gamma), gives smooth transition for Kp =1.0; 

        #set an empty Phi_singular matrix, populate if there were any adjusted
        #condition numbers below the threshold
        Phi_singular = np.zeros(U.shape) #initialize the singular projection matrix  

        if verbose == True:
            print("set_empty_bool:", set_empty_bool)
        if set_empty_bool == False:
            #construct the new U, as there were singular directions
            U_new_sing = np.matrix(np.concatenate(U_new_sing,axis=0)).T
            Phi_mat = np.matrix(np.diag(Phi))
            # if position_only == True:
            #     gains = np.array([singular_direction_gain_position]*3, dtype=float)
            # else:
            #     gains = np.array([singular_direction_gain_position]*3 + [singular_direction_gain_angular]*3, dtype=float)
            # Kp_singular = np.diag(gains)
            # Now put it all together:
            Phi_singular = U_new_sing @ Phi_mat @ U_new_sing.T #@ Kp_singular
            if verbose == True:
                # print("Kp_singular:", Kp_singular)            
                print("Phi_mat shape:", Phi_mat.shape, "Phi_mat:", Phi_mat)
                print("U_new_sing shape:", U_new_sing.shape, "U_new_sing:", U_new_sing)
        
        #Obtain psuedo-inverse of the safety jacobian and the projection jacobian
        J_safety_pinv = np.linalg.pinv(J_safety)
        J_proj_pinv = np.linalg.pinv(J_proj)

        if set_empty_bool == False:
            J_parse = J_safety_pinv @ J_proj @ J_proj_pinv + J_safety_pinv @ Phi_singular
        else:
            J_parse = J_safety_pinv @ J_proj @ J_proj_pinv

        #Publish the JParse ellipses
        ellipse_dict = {"J_safety_u": U, "J_safety_s": S_new_safety, "J_proj_u": U_new_proj, "J_proj_s": S_new_proj, "J_singular_u": U_new_sing, "J_singular_s": Phi}     
        if internal_marker_flag == True:
            #this is internal for jparse marker display
            return ellipse_dict
        
        if publish_jparse_ellipses == True:
            #only shows position ellipses
            self.publish_position_Jparse_ellipses(q=q, gamma=gamma, jac_nullspace_bool=False, singular_direction_gain_position=singular_direction_gain_position, singular_direction_gain_angular=singular_direction_gain_angular, verbose=verbose, publish_jparse_ellipses=publish_jparse_ellipses, end_effector_pose=end_effector_pose)

        if jac_nullspace_bool == True and not safety_only and not safety_projection_only:
            #Find the nullspace of the jacobian
            J_safety_nullspace = np.eye(J_safety.shape[1]) - J_safety_pinv @ J_safety

            return J_parse, J_safety_nullspace
        
        if safety_only == True:
            return J_safety_pinv
        
        if safety_projection_only == True:
            return J_safety_pinv @ J_proj @ J_proj_pinv

        return J_parse

    '''
    The following are only useful for the dynamics terms (e.g. torque control)
    They are not required to obtain the JParse matrix
    '''
    def publish_position_Jparse_ellipses(self, q=[], gamma=0.1, jac_nullspace_bool = False , singular_direction_gain_position=1, singular_direction_gain_angular=1, verbose=False, publish_jparse_ellipses=False, end_effector_pose=None):
        """ 
        this function displays the key ellipses for the JParse
        """
        #obtain the key matricies
        ellipse_marker_array = MarkerArray()
       
        ellipse_dict = self.JParse(q=q, gamma=gamma, position_only=True, jac_nullspace_bool=jac_nullspace_bool, singular_direction_gain_position=singular_direction_gain_position, singular_direction_gain_angular=singular_direction_gain_angular, verbose=verbose, publish_jparse_ellipses=False, internal_marker_flag=True, end_effector_pose=end_effector_pose)
        frame_id = self.base_link
        #add safety jacobian
        J_safety_marker = self.generate_jparse_ellipses(mat_type="J_safety", U_mat=ellipse_dict["J_safety_u"], S_vect=ellipse_dict["J_safety_s"], marker_ns="J_safety", end_effector_pose=end_effector_pose, frame_id=frame_id)
        ellipse_marker_array.markers.append(J_safety_marker[0])
        #if there are feasible directions, add J_proj
        if len(ellipse_dict["J_proj_s"]) > 0:
            #Pass in the full U to ensure the ellipse shows up at all, we will handle with approximate scaling for singular directions (make them 0.001)
            J_proj_marker = self.generate_jparse_ellipses(mat_type="J_proj", U_mat=ellipse_dict["J_safety_u"], S_vect=ellipse_dict["J_proj_s"], marker_ns="J_proj", end_effector_pose=end_effector_pose, frame_id=frame_id)
            ellipse_marker_array.markers.append(J_proj_marker[0])
        #if there are singular directions, add them        
        J_singular_marker = self.generate_jparse_ellipses(mat_type="J_singular", U_mat=ellipse_dict["J_singular_u"], S_vect=ellipse_dict["J_singular_s"], end_effector_pose=end_effector_pose, frame_id=frame_id)
        if len(J_singular_marker) > 0:
            for idx in range(len(J_singular_marker)):
                ellipse_marker_array.markers.append(J_singular_marker[idx]) 
        self.marker_pub.publish(ellipse_marker_array)

    def generate_jparse_ellipses(self, mat_type=None, U_mat=None, S_vect=None, marker_ns="ellipse" , frame_id="base_link", end_effector_pose=None):
        #This function takes in the singular value directions and plots ellipses or vectors
        Marker_list = []
        pose = PoseStamped()
        if mat_type in ["J_safety", "J_proj"]:
            marker = Marker()
            #Create the marker
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = marker_ns
            if mat_type == "J_safety":
                marker.id = 0
                marker_ns = "J_safety"
            elif mat_type == "J_proj":
                marker.id = 1
                marker_ns = "J_proj"
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            #set the position of the marker
            marker.pose.position.x = end_effector_pose.pose.position.x
            marker.pose.position.y = end_effector_pose.pose.position.y
            marker.pose.position.z = end_effector_pose.pose.position.z

            #set the scale based on singular values (adjust factor if needed); in practice, the ellipsoid should flatten, but for easy visualization, we make these dimensions very very small            
            ellipsoid_scale = 0.25
            marker.scale.x = 0.001
            marker.scale.y = 0.001
            marker.scale.z = 0.001
            if len(S_vect) == 1:
                marker.scale.x = ellipsoid_scale*np.max([S_vect[0], 0.001]) 
            elif len(S_vect) == 2:
                marker.scale.x = ellipsoid_scale*np.max([S_vect[0], 0.001]) 
                marker.scale.y = ellipsoid_scale*np.max([S_vect[1], 0.001])
            elif len(S_vect) >= 3:
                marker.scale.x = ellipsoid_scale*np.max([S_vect[0], 0.001]) 
                marker.scale.y = ellipsoid_scale*np.max([S_vect[1], 0.001])
                marker.scale.z = ellipsoid_scale*np.max([S_vect[2], 0.001])
            
            # Convert U (rotation matrix) to quaternion
            q = self.rotation_matrix_to_quaternion(U_mat)
            marker.pose.orientation = q

            # Set color (RGBA)
            if mat_type == "J_safety":
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.7 #transparency
            elif mat_type == "J_proj":
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 0.7 #transparency
            Marker_list.append(marker)

        elif mat_type == "J_singular":
            
            for idx in range(len(S_vect)):
                #Create the marker
                marker = Marker()
                marker_ns = "J_singular"
                marker.header.frame_id = frame_id
                marker.header.stamp = rospy.Time.now()
                marker.ns = marker_ns
                marker.id = idx+2
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                marker.lifetime = rospy.Duration(0.1)  # or rclpy.duration.Duration(seconds=0.1) in ROS 2

                #Arrow start point
                start_point = Point()
                start_point.x = end_effector_pose.pose.position.x
                start_point.y = end_effector_pose.pose.position.y
                start_point.z = end_effector_pose.pose.position.z

                #arrow end point
                arrow_scale = 1.0
                end_point = Point()
                # if points away from origin 
                if (end_effector_pose.pose.position.x*U_mat[0,idx] + end_effector_pose.pose.position.y*U_mat[1,idx] + end_effector_pose.pose.position.z*U_mat[2,idx]) < 0:
                    arrow_x = U_mat[0,idx]
                    arrow_y = U_mat[1,idx]
                    arrow_z = U_mat[2,idx]
                else:
                    arrow_x = -U_mat[0,idx]
                    arrow_y = -U_mat[1,idx]
                    arrow_z = -U_mat[2,idx]
                end_point.x = end_effector_pose.pose.position.x + arrow_scale*arrow_x*np.abs(S_vect[idx])
                end_point.y = end_effector_pose.pose.position.y + arrow_scale*arrow_y*np.abs(S_vect[idx])
                end_point.z = end_effector_pose.pose.position.z + arrow_scale*arrow_z*np.abs(S_vect[idx])

                marker.points.append(start_point)
                marker.points.append(end_point)

                # Scale (arrow thickness and head size)
                marker.scale.x = 0.01  # Shaft diameter
                marker.scale.y = 0.05   # Head diameter
                marker.scale.z = 0.05   # Head length

                # Set color based on principal axis
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0  # Opaque

                Marker_list.append(marker)

        return Marker_list 

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a 3x3 rotation matrix to a quaternion.
        :param R: 3x3 rotation matrix
        :return: geometry_msgs/Quaternion
        """
        q = Quaternion()
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        
        # Normalize quaternion
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        q.x = qx / norm
        q.y = qy / norm
        q.z = qz / norm
        q.w = qw / norm
        return q

    def cart_inertia(self, q=[]):
        """
        This is not needed for the main JParse class, but is included here for reference.
        """
        H = self.inertia(q)
        J = self.jacobian(q)
        J_pinv = np.linalg.pinv(J)
        J_pinv_transpose = J_pinv.T
        return J_pinv_transpose @ H @ J_pinv

    def inertia(self, q=[]):
        """
        This is not needed for the main JParse class, but is included here for reference.
        """
        h_kdl = kdl.JntSpaceInertiaMatrix(self.num_joints)
        self._dyn_kdl.JntToMass(joint_list_to_kdl(q), h_kdl)
        return kdl_to_mat(h_kdl)

    def coriolis(self,q=[], qdot=[]):
        """
        This is not needed for the main JParse class, but is included here for reference.
        """
        q = q #list
        qdot = qdot #list
        q_cori = [0.0 for idx in range(len(q))]
        q_kdl = joint_list_to_kdl(q)
        qdot_kdl = joint_list_to_kdl(qdot)
        q_cori_kdl = joint_list_to_kdl(q_cori)
        self._dyn_kdl.JntToCoriolis(q_kdl, qdot_kdl, q_cori_kdl)
        q_cori_kdl = np.array([q_cori_kdl[i] for i in range(q_cori_kdl.rows())])
        q_cori_kdl = np.matrix(q_cori_kdl).T
        return q_cori_kdl

    def gravity(self,q=[]):
        """
        This is not needed for the main JParse class, but is included here for reference.
        """
        q_grav = [0.0 for idx in range(len(q))]
        q_kdl = joint_list_to_kdl(q)
        q_grav_kdl = joint_list_to_kdl(q_grav)
        self._dyn_kdl.JntToGravity(q_kdl,q_grav_kdl)
        q_grav_kdl = np.array([q_grav_kdl[i] for i in range(q_grav_kdl.rows())])
        q_grav_kdl = np.matrix(q_grav_kdl).T
        return q_grav_kdl

    '''
    Baseline implementations for comparison
    '''
    def jacobian_damped_least_squares(self, q=[], damping=0.01, jac_nullspace_bool=False):
        """
        COMPARISON METHOD (not used in JPARSE)
        This function computes the damped least squares pseudo-inverse of the Jacobian matrix for the given joint configuration.
        """
        J = self.jacobian(q)
        J_dls = np.linalg.inv(J.T @ J + damping**2 * np.eye(J.shape[1])) @ J.T
        if jac_nullspace_bool == False:
            return J_dls
        J_dls_nullspace = np.eye(J.shape[1]) - J_dls @ J
        return J_dls, J_dls_nullspace

    def jacobian_transpose(self, q=[]):
        """
        COMPARISON METHOD (not used in JPARSE)
        This function computes the transpose of the Jacobian matrix for the given joint configuration.
        """
        J = self.jacobian(q)
        J_transpose = J.T
        J_transpose_nullspace = np.eye(J_transpose.shape[0]) - J_transpose @ np.linalg.pinv(J_transpose)
        return J_transpose, J_transpose_nullspace
    
    def jacobian_projection(self, q=[], gamma=0.1, jac_nullspace_bool=False):
        """
        This function computes the projection of the Jacobian matrix onto feasible (non-singular) directions for the given joint configuration.
        """
        J = self.jacobian(q)
        #Perform the SVD decomposition of the jacobian
        U, S, Vt = np.linalg.svd(J)
        #Find the adjusted condition number
        sigma_max = np.max(S)
        #Find the projection Jacobian
        U_new_proj = []
        S_new_proj = []
        for col in range(len(S)):
            if S[col] > gamma*sigma_max:
                U_new_proj.append(np.matrix(U[:,col]).T)
                S_new_proj.append(S[col])
            else:
                rospy.loginfo("Singular row found during projection")
                print("the ratio of s_i/s_max is:", S[col]/sigma_max)

        U_new_proj = np.concatenate(U_new_proj,axis=0).T
        J_proj = self.svd_compose(U_new_proj, S_new_proj, Vt)
        #if there is no nullspace, return the projection jacobian
        if jac_nullspace_bool == False:
            return J_proj
        J_proj_nullspace = np.eye(J_proj.shape[1]) -   np.linalg.pinv(J_proj)@J_proj
        return J_proj, J_proj_nullspace


def main():
    rospy.init_node('jparse_test', anonymous=True)
    jparse = JParseClass()
    q = [0, 0, 0, 0, 0, 0]
    print("JParse:", jparse.JParse(q=q))
    print("JParse:", jparse.JParse(q=q, position_only=True))
    print("Jacobian:", jparse.jacobian(q=q))
    print("Jacobian:", jparse.jacobian(q=q, position_only=True))
    print("Jacobian Pseudo Inverse:", jparse.JacobianPseudoInverse(q=q))
    print("Jacobian Pseudo Inverse:", jparse.JacobianPseudoInverse(q=q, position_only=True))
    print("Jacobian Damped Least Squares:", jparse.jacobian_damped_least_squares(q=q))
    print("Jacobian Damped Least Squares:", jparse.jacobian_damped_least_squares(q=q, jac_nullspace_bool=True))
    print("Jacobian Transpose:", jparse.jacobian_transpose(q=q))

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

