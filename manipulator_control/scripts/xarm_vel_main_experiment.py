#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import math
import scipy
import time

import PyKDL as kdl
from urdf_parser_py.urdf import Robot
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_kinematics import kdl_to_mat
from pykdl_utils.kdl_kinematics import joint_kdl_to_list
from pykdl_utils.kdl_kinematics import joint_list_to_kdl

import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, Vector3, TwistStamped
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState
import tf
from tf import TransformerROS
import tf2_ros
from tf.transformations import quaternion_from_euler, quaternion_matrix

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

try:
    from xarm.wrapper import XArmAPI
except ImportError: 
    print("xarm package not installed, skipping xarm import")

from manipulator_control import jparse_cls 

class ArmController:
    def __init__(self):
        rospy.init_node('arm_controller', anonymous=True)
              
        self.base_link = rospy.get_param('/base_link_name', 'link_base') #defaults are for Kinova gen3
        self.end_link = rospy.get_param('/end_link_name', 'link_eef')

        self.is_sim = rospy.get_param('~is_sim', False) #boolean to control if the robot is in simulation or not

        self.phi_gain = rospy.get_param('~phi_gain', 10) #gain for the phi term in the JParse method
        self.phi_gain_position = rospy.get_param('~phi_gain_position', 15) #gain for the phi term in the JParse method
        self.phi_gain_angular = rospy.get_param('~phi_gain_angular', 15) #gain for the phi term in the JParse method
        
        self.jparse_gamma = rospy.get_param('~jparse_gamma', 0.1) #gamma for the JParse method

        self.use_space_mouse = rospy.get_param('~use_space_mouse', False) #boolean to control if the space mouse is used or not
        self.use_space_mouse_jparse = rospy.get_param('~use_space_mouse_jparse', False) #boolean to control if the space mouse is used or not

        self.space_mouse_command = np.array([0,0,0,0,0,0]).T

        if self.is_sim == False:
            self.robot_ip = rospy.get_param('~robot_ip', '192.168.1.233 ')
            self.arm = XArmAPI(self.robot_ip)
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(state=0)
            time.sleep(1)

            # set joint velocity control mode
            self.arm.set_mode(4)
            self.arm.set_state(0)
            time.sleep(1)

        #Set parameters
        self.position_only = rospy.get_param('~position_only', False) #boolean to control only position versus full pose

        #choose the method to use
        self.method = rospy.get_param('~method', 'JParse') #options are JParse, JacobianPseudoInverse, JacobianDampedLeastSquares, JacobianProjection, JacobianDynamicallyConsistentInverse
        
        self.show_jparse_ellipses = rospy.get_param('~show_jparse_ellipses', False) #boolean to control if the jparse ellipses are shown or not

        self.joint_states = None
        self.target_pose = None
        self.jacobian_calculator = jparse_cls.JParseClass(base_link=self.base_link, end_link=self.end_link)

        #get tf listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tfros = tf.TransformerROS()
        self.tf_timeout = rospy.Duration(1.0)

        if self.is_sim == True:
            #if we are in simulation, use the joint_states and target_pose topics
            joint_states_topic = '/xarm/joint_states'
        else:
            joint_states_topic = '/joint_states'

        rospy.Subscriber(joint_states_topic, JointState, self.joint_states_callback)
        rospy.Subscriber('/target_pose', PoseStamped, self.target_pose_callback)
        rospy.Subscriber('/robot_action', TwistStamped, self.space_mouse_callback)
        
        
        self.manip_measure_pub = rospy.Publisher('/manip_measure', Float64, queue_size=10)
        self.inverse_cond_number = rospy.Publisher('/inverse_cond_number', Float64, queue_size=10)
        #have certain messages to store raw error
        self.pose_error_pub = rospy.Publisher('/pose_error', PoseStamped, queue_size=10)
        self.position_error_pub = rospy.Publisher('/position_error', Vector3, queue_size=10)
        self.orientation_error_pub = rospy.Publisher('/orientation_error', Vector3, queue_size=10)
        #have certain messages to store control error
        self.pose_error_control_pub = rospy.Publisher('/pose_error_control', PoseStamped, queue_size=10)
        self.position_error_control_pub = rospy.Publisher('/position_error_control', Vector3, queue_size=10)
        self.orientation_control_error_pub = rospy.Publisher('/orientation_error_control', Vector3, queue_size=10)
        #publish the current end effector pose and target pose
        self.current_end_effector_pose_pub = rospy.Publisher('/current_end_effector_pose', PoseStamped, queue_size=10)
        self.current_target_pose_pub = rospy.Publisher('/current_target_pose', PoseStamped, queue_size=10)

        joint_command_topic = rospy.get_param('~joint_command_topic', '/xarm/xarm7_velo_traj_controller/command')
        self.joint_vel_pub = rospy.Publisher(joint_command_topic, JointTrajectory, queue_size=10)
        # Define the joint names
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        # Initialize the current positions
        self.current_positions = [0.0] * len(self.joint_names)

        self.rate = rospy.Rate(50)  # 10 Hz
        #home the robot
        try:
            self.velocity_home_robot()
        except rospy.ROSInterruptException:
            rospy.loginfo("ROS Interrupt Exception")
            pass
        finally:
            # Clean up resources if needed
            rospy.loginfo("Shutting down the control loop")
            joint_zero_velocities = [0.0] * len(self.joint_names)
            self.command_joint_velocities(joint_zero_velocities)
        #now run the control loop
        self.control_loop()

    def rad2deg(self, q):
        return q/math.pi*180.0

    def joint_states_callback(self, msg):
        """
        Callback function for the joint_states topic. This function will be called whenever a new message is received
        on the joint_states topic. The message is a sensor_msgs/JointState message, which contains the current joint
        and the corresponding joint velocities and efforts. This function will extract the joint positions, velocities, 
        and efforts and return them as lists.
        """
        self.joint_states = msg

    def space_mouse_callback(self, msg):
        """
        Callback function for the space_mouse topic. This function will be called whenever a new message is received
        on the space_mouse topic. The message is a geometry_msgs/TwistStamped message, which contains the current
        velocity of the end effector. This function will extract the linear and angular velocities and return them as
        lists.
        """
        space_mouse_command = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z, msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]) 
        position_velocity = space_mouse_command[:3]
        angular_velocity = space_mouse_command[3:]
        position_velocity_norm = np.linalg.norm(position_velocity)
        angular_velocity_norm = np.linalg.norm(angular_velocity)
        if position_velocity_norm > 0.05:
            position_velocity = position_velocity / position_velocity_norm * 0.05
        self.space_mouse_command = np.array([position_velocity[0], position_velocity[1],position_velocity[2],angular_velocity[0],angular_velocity[1],angular_velocity[2]])        #check if norm of the space mouse command is greater than 0.05, if so normalize it to this value

        #ensure we can get into the while loop
        if self.use_space_mouse == True:
            self.target_pose = PoseStamped() #set this to get in a loop


    def target_pose_callback(self, msg):
        """
        Callback function for the target_pose topic. This function will be called whenever a new message is received
        on the target_pose topic. The message is a geometry_msgs/PoseStamped message, which contains the target pose
        """
        self.target_pose = msg
        self.current_target_pose_pub.publish(self.target_pose) #if target pose is paused manually, this allows us to track the current target pose seen by the script

    def EndEffectorPose(self, q):
        """
        This function computes the end-effector pose given the joint configuration q.
        """
        current_pose = PoseStamped()
        current_pose.header.frame_id = self.base_link
        current_pose.header.stamp = rospy.Time.now()
        try:
            trans = self.tfBuffer.lookup_transform(self.base_link, self.end_link, rospy.Time(), timeout=self.tf_timeout)
            current_pose.pose.position.x = trans.transform.translation.x
            current_pose.pose.position.y = trans.transform.translation.y
            current_pose.pose.position.z = trans.transform.translation.z
            current_pose.pose.orientation.x = trans.transform.rotation.x
            current_pose.pose.orientation.y = trans.transform.rotation.y
            current_pose.pose.orientation.z = trans.transform.rotation.z
            current_pose.pose.orientation.w = trans.transform.rotation.w
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("TF lookup failed")
            rospy.logerr("failed to lookup transform from %s to %s", self.base_link, self.end_link)
        self.current_end_effector_pose_pub.publish(current_pose)
        return current_pose

    def EndEffectorVelocity(self, q, dq):
        """
        This function computes the end-effector velocity given the joint configuration q and joint velocities dq.
        """
        J = self.jacobian_calculator.jacobian(q)
        J = np.array(J)
        dq = np.array(dq)
        dx = np.dot(J, dq)
        return dx

    def rotation_matrix_to_axis_angle(self,R):
        """
        Converts a rotation matrix to an axis-angle vector.
        
        Parameters:
            R (numpy.ndarray): A 3x3 rotation matrix.
        
        Returns:
            numpy.ndarray: Axis-angle vector (3 elements).
        """
        if not (R.shape == (3, 3) and np.allclose(np.dot(R.T, R), np.eye(3)) and np.isclose(np.linalg.det(R), 1)):
            raise ValueError("Input matrix must be a valid rotation matrix.")
        
        # Calculate the angle of rotation
        angle = np.arccos((np.trace(R) - 1) / 2)

        if np.isclose(angle, 0):  # No rotation
            return np.zeros(3)

        if np.isclose(angle, np.pi):  # 180 degrees rotation
            # Special case for 180-degree rotation
            # Extract the axis from the diagonal of R
            axis = np.sqrt((np.diag(R) + 1) / 2)
            # Adjust signs based on the matrix off-diagonals
            axis[0] *= np.sign(R[2, 1] - R[1, 2])
            axis[1] *= np.sign(R[0, 2] - R[2, 0])
            axis[2] *= np.sign(R[1, 0] - R[0, 1])
            return axis * angle

        # General case
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(angle))

        return axis * angle

    def pose_error(self, current_pose, target_pose, error_norm_max = 0.05, control_pose_error = True):
        """
        This function computes the error between the current pose and the target pose.
        """
        #Compute the position error
        position_error = np.array([target_pose.pose.position.x - current_pose.pose.position.x,
                                   target_pose.pose.position.y - current_pose.pose.position.y,
                                   target_pose.pose.position.z - current_pose.pose.position.z])
        #if the norm of the position error is greater than the maximum allowed, scale it down
        if control_pose_error == True:
            if np.linalg.norm(position_error) > error_norm_max:
                position_error = position_error / np.linalg.norm(position_error) * error_norm_max

        #convert the quaternion in posestamped to list of quaternions then pass this into quaternion_matrix to get the rotation matrix. Access only the rotation part of the matrix
        goal_rotation_matrix = quaternion_matrix([target_pose.pose.orientation.x, target_pose.pose.orientation.y, target_pose.pose.orientation.z, target_pose.pose.orientation.w])[:3,:3]
        current_rotation_matrix = quaternion_matrix([current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w])[:3,:3]
        #Compute the orientation error
        R_error= np.dot(goal_rotation_matrix, np.linalg.inv(current_rotation_matrix))
        # Extract the axis-angle lie algebra vector from the rotation matrix
        angle_error = self.rotation_matrix_to_axis_angle(R_error)
        # Return the position and orientation error
        return position_error, angle_error

    def axis_angle_to_rotation_matrix(self, axis_angle):
        """
        Converts an axis-angle vector to a rotation matrix.
        
        Parameters:
            axis_angle (numpy.ndarray): Axis-angle vector (3 elements).
        
        Returns:
            numpy.ndarray: A 3x3 rotation matrix.
        """
        # Extract the axis and angle
        axis = axis_angle / np.linalg.norm(axis_angle)
        angle = np.linalg.norm(axis_angle)
        
        # Compute the skew-symmetric matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        # Compute the rotation matrix using the Rodrigues' formula
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        return R
    
    def rotation_matrix_to_quaternion(self, R):
        """
        Converts a rotation matrix to a quaternion.
        
        Parameters:
            R (numpy.ndarray): A 3x3 rotation matrix.
        
        Returns:
            numpy.ndarray: A 4-element quaternion.
        """
        if not (R.shape == (3, 3) and np.allclose(np.dot(R.T, R), np.eye(3)) and np.isclose(np.linalg.det(R), 1)):
            raise ValueError("Input matrix must be a valid rotation matrix.")
        
        # Compute the quaternion using the method from the book
        q = np.zeros(4)
        q[0] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
        q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
        q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])
        
        return q

    def publish_pose_errors(self, position_error, angular_error, control_pose_error = True):
        """
        Publishes the position and orientation errors as ROS messages.
        """
        pose_error_msg = PoseStamped()
        pose_error_msg.header.frame_id = self.base_link
        pose_error_msg.header.stamp = rospy.Time.now()
        pose_error_msg.pose.position.x = position_error[0]
        pose_error_msg.pose.position.y = position_error[1]
        pose_error_msg.pose.position.z = position_error[2]
        #for completeness, convert the axis angle to quaternion
        Rmat = self.axis_angle_to_rotation_matrix(angular_error)
        quat_from_Rmat = self.rotation_matrix_to_quaternion(Rmat)
        # quat_from_Rmat = tf.transformations.quaternion_from_matrix(Rmat)
        pose_error_msg.pose.orientation.x = quat_from_Rmat[0]
        pose_error_msg.pose.orientation.y = quat_from_Rmat[1]
        pose_error_msg.pose.orientation.z = quat_from_Rmat[2]
        pose_error_msg.pose.orientation.w = quat_from_Rmat[3]
        #publish the axis-angle orientation error
        orientation_error_msg = Vector3()
        orientation_error_msg.x = angular_error[0]
        orientation_error_msg.y = angular_error[1]
        orientation_error_msg.z = angular_error[2]
        #publish the position error
        position_error_msg = Vector3()
        position_error_msg.x = position_error[0]
        position_error_msg.y = position_error[1]
        position_error_msg.z = position_error[2]

        #Now publish
        if control_pose_error == True:
            self.position_error_control_pub.publish(position_error_msg)
            self.pose_error_control_pub.publish(pose_error_msg)
            self.orientation_control_error_pub.publish(orientation_error_msg)
        else:
            self.pose_error_pub.publish(pose_error_msg)
            self.position_error_pub.publish(position_error_msg)
            self.orientation_error_pub.publish(orientation_error_msg)


    def control_loop(self):
        """
        This function implements the control loop for the arm controller.
        """

        while not rospy.is_shutdown():

            if (self.joint_states and self.target_pose and not self.use_space_mouse) or (self.use_space_mouse and self.joint_states and self.space_mouse_command):
                t = rospy.Time.now()
                # obtain the current joints
                q = []
                dq = []
                effort = []
                for joint_name in self.joint_names:
                    idx = self.joint_states.name.index(joint_name)
                    q.append(self.joint_states.position[idx])
                    dq.append(self.joint_states.velocity[idx])
                    effort.append(self.joint_states.effort[idx])  
                self.current_positions = q
                # Calculate the JParsed Jacobian
                
                method = self.method #set by parameter, can be set from launch file
                rospy.loginfo("Method being used: %s", method)
                if method == "JacobianPseudoInverse":
                    #this is the traditional pseudo-inverse method for the jacobian
                    J_method, J_nullspace = self.jacobian_calculator.JacobianPseudoInverse(q=q, position_only=self.position_only, jac_nullspace_bool=True)
                elif method == "JParse":
                    # The JParse method takes in the joint angles, gamma, position_only, and singular_direction_gain
                    if self.show_jparse_ellipses == True:
                        J_method, J_nullspace = self.jacobian_calculator.JParse(q=q, gamma=self.jparse_gamma, position_only=self.position_only, singular_direction_gain_position=self.phi_gain_position, singular_direction_gain_angular=self.phi_gain_angular,  jac_nullspace_bool=True, publish_jparse_ellipses=True, end_effector_pose=self.EndEffectorPose(q), verbose=False)                        
                    else:
                        J_method, J_nullspace = self.jacobian_calculator.JParse(q=q, gamma=self.jparse_gamma, position_only=self.position_only, singular_direction_gain_position=self.phi_gain_position,singular_direction_gain_angular=self.phi_gain_angular, jac_nullspace_bool=True)
                elif method == "JacobianDampedLeastSquares":
                    J_method, J_nullspace = self.jacobian_calculator.jacobian_damped_least_squares(q=q, damping=0.1, jac_nullspace_bool=True) #dampening of 0.1 works very well, 0.8 shows clear error
                elif method == "JacobianProjection":
                    J_proj, J_nullspace = self.jacobian_calculator.jacobian_projection(q=q, gamma=0.1, jac_nullspace_bool=True)
                    J_method = np.linalg.pinv(J_proj)
                elif method == "JacobianSafety":
                    # The JParse method takes in the joint angles, gamma, position_only, and singular_direction_gain
                    if self.show_jparse_ellipses == True:
                        J_method, J_nullspace = self.jacobian_calculator.JParse(q=q, gamma=self.jparse_gamma, position_only=self.position_only, singular_direction_gain_position=self.phi_gain_position, singular_direction_gain_angular=self.phi_gain_angular,  jac_nullspace_bool=True, publish_jparse_ellipses=True, end_effector_pose=self.EndEffectorPose(q), verbose=False, safety_only=True)                        
                    else:
                        J_method, J_nullspace = self.jacobian_calculator.JParse(q=q, gamma=self.jparse_gamma, position_only=self.position_only, singular_direction_gain_position=self.phi_gain_position,singular_direction_gain_angular=self.phi_gain_angular, jac_nullspace_bool=True, safety_only=True)
                elif method == "JacobianSafetyProjection":
                    # The JParse method takes in the joint angles, gamma, position_only, and singular_direction_gain
                    if self.show_jparse_ellipses == True:
                        J_method, J_nullspace = self.jacobian_calculator.JParse(q=q, gamma=self.jparse_gamma, position_only=self.position_only, singular_direction_gain_position=self.phi_gain_position, singular_direction_gain_angular=self.phi_gain_angular,  jac_nullspace_bool=True, publish_jparse_ellipses=True, end_effector_pose=self.EndEffectorPose(q), verbose=False, safety_projection_only=True)                        
                    else:
                        J_method, J_nullspace = self.jacobian_calculator.JParse(q=q, gamma=self.jparse_gamma, position_only=self.position_only, singular_direction_gain_position=self.phi_gain_position,singular_direction_gain_angular=self.phi_gain_angular, jac_nullspace_bool=True, safety_projection_only=True)


                manip_measure = self.jacobian_calculator.manipulability_measure(q)
                self.manip_measure_pub.publish(manip_measure)
                inverse_cond_number = self.jacobian_calculator.inverse_condition_number(q)
                self.inverse_cond_number.publish(inverse_cond_number)
                rospy.loginfo("Manipulability measure: %f", manip_measure)
                rospy.loginfo("Inverse condition number: %f", inverse_cond_number)
                #Calculate the delta_x (task space error)
                target_pose = self.target_pose
                current_pose = self.EndEffectorPose(q)
                position_error, angular_error = self.pose_error(current_pose, target_pose, control_pose_error=False)
                # log the pose error
                self.publish_pose_errors(position_error, angular_error, control_pose_error=False)
                #now find error for control (max error norm)
                if self.is_sim == False:
                    #real world limits
                    error_norm_max = 0.10
                else:
                    #simulation limits
                    error_norm_max = 1.0
                position_error, angular_error = self.pose_error(current_pose, target_pose, error_norm_max = error_norm_max, control_pose_error=True)                
                #move in nullspace towards nominal pose (for xarm its 0 joint angle for Xarm7)
                if self.is_sim == True:
                    kp_gain = 2.0
                else:
                    # kp_gain = 1.0
                    kp_gain = 3.0

                if self.is_sim == True:
                    nominal_motion_nullspace = np.matrix([-v*kp_gain for v in q]).T #send to home which is 0 joint position for all joints
                else:
                    # nominal motion nullspace which checks if q magnitude is below threshold and chooses the minimum
                    null_space_angle_rate = 0.6

                    nominal_motion_nullspace = np.matrix([np.sign(-v*kp_gain)*np.min([np.abs(-v*kp_gain),null_space_angle_rate]) for v in q]).T #send to home which is 0 joint position for all joints
                # log the pose error
                self.publish_pose_errors(position_error, angular_error, control_pose_error=True)

                # Calculate and command the joint velocities
                if self.position_only == True:
                    rospy.loginfo("Position only control")
                    position_error = np.matrix(position_error).T
                    if self.is_sim == True:
                        #use these gains only in simulation!
                        kp_gain = 10.0
                    else:
                        kp_gain = 3.0
                    Kp = np.diag([kp_gain, kp_gain, kp_gain])  # Proportional gain matrix

                    task_vector = Kp @ position_error 
                    joint_velocities = J_method @ task_vector + J_nullspace @ nominal_motion_nullspace
                else:
                    # realworld gains (tested)
                    kp_gain_position = 1.0
                    kp_gain_orientation = 1.0
                    if self.is_sim == True:
                        #use these gains only in simulation!
                        kp_gain_position = 10.0
                        kp_gain_orientation = 10.0
                    Kp_position = np.diag([kp_gain_position, kp_gain_position, kp_gain_position])  # Proportional gain matrix
                    Kp_orientation = np.diag([kp_gain_orientation, kp_gain_orientation, kp_gain_orientation])
                    task_position = Kp_position @ np.matrix(position_error).T
                    task_orientation = Kp_orientation @ np.matrix(angular_error).T
                    full_pose_error = np.matrix(np.concatenate((task_position.T, task_orientation.T),axis=1)).T
                    if self.use_space_mouse == False:
                        joint_velocities = J_method @ full_pose_error + J_nullspace @ nominal_motion_nullspace
                    if self.use_space_mouse == True:
                        #use the space mouse command to control the joint velocities
                        space_mouse_command = np.matrix(self.space_mouse_command).T
                        #now add this to the joint velocities
                        joint_velocities = J_method @ space_mouse_command + J_nullspace @ nominal_motion_nullspace
                        #check this
                joint_velocities_list = np.array(joint_velocities).flatten().tolist()
                # command the joint velocities
                self.command_joint_velocities(joint_velocities_list) #this commands the joint velocities
            self.rate.sleep()
            rospy.loginfo("Control loop running")


    def velocity_home_robot(self):
        """
        This function commands the robot to the home position using joint velocities.
        """
        # do the following in a loop for 5 seconds:
        t_start = rospy.Time.now()
        duration = 0.0
        while not rospy.is_shutdown():
            if duration >= 5.0:
                break
            rospy.loginfo("Homing the robot")
            if self.joint_states:
                duration = (rospy.Time.now() - t_start).to_sec()
                q = []
                dq = []
                for joint_name in self.joint_names:
                    idx = self.joint_states.name.index(joint_name)
                    q.append(self.joint_states.position[idx])
                    dq.append(self.joint_states.velocity[idx])
                self.current_positions = q
                # Command the joint velocities
                if self.is_sim == True:
                    kp_gain = 10.0
                else:
                    kp_gain = 0.5
                q_home = [-0.03142359480261803, -0.5166178941726685, 0.12042707949876785, 0.9197863936424255, -0.03142168000340462, 1.4172008037567139, 0.03490765020251274]
                #now find the error between the current position and the home position and use joint velocities to move towards the home position
                joint_velocities_list = kp_gain * (np.array(q_home) - np.array(q))
                
                self.command_joint_velocities(joint_velocities_list)


    def command_joint_velocities(self, joint_vel_list):
        """
        This function commands the joint velocities to the robot using the appropriate ROS message type.
        """
        if self.is_sim:
            # Create the JointTrajectory message
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_names

            # Create a trajectory point
            point = JointTrajectoryPoint()

            # Use current positions
            point.positions = self.current_positions

            # Set velocities
            #make velocity negative because Xarm has cw positive direction for joint velocities
            joint_vel_list = [-v for v in joint_vel_list]
            point.velocities = joint_vel_list #this is negative because Xarm has cw positive direction for joint velocities

            # Set accelerations to zero
            point.accelerations = [0.0] * len(self.joint_names)

            # Effort (optional; set to None or skip if not needed)
            point.effort = [0.0] * len(self.joint_names)

            # Set the time from start
            point.time_from_start = rospy.Duration(0.1)  # Duration in seconds

            # Add the point to the trajectory
            trajectory_msg.points.append(point)

            # Publish the trajectory
            trajectory_msg.header.stamp = rospy.Time.now()  # Update timestamp
            self.joint_vel_pub.publish(trajectory_msg)
        else:
            # this is on the real robot, directly send joint velociteies
            # Send joint velocities to the arm
            rospy.loginfo("Joint velocities: %s", joint_vel_list)
            if self.use_space_mouse_jparse:
                self.arm.vc_set_joint_velocity(joint_vel_list, is_radian=True)


if __name__ == '__main__':
    try:
        ArmController()
    except rospy.ROSInterruptException:
        pass

