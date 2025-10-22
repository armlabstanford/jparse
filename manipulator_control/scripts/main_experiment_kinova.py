#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import math
import scipy
import time 
import cv2

import PyKDL as kdl
from urdf_parser_py.urdf import Robot
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
from pykdl_utils.kdl_kinematics import kdl_to_mat
from pykdl_utils.kdl_kinematics import joint_kdl_to_list
from pykdl_utils.kdl_kinematics import joint_list_to_kdl

import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, Vector3, TwistStamped, TransformStamped, PoseWithCovarianceStamped
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState
import tf
from tf import TransformerROS
import tf2_ros
from tf.transformations import quaternion_from_matrix,  quaternion_matrix

# import april tag detection array
from apriltag_ros.msg import AprilTagDetectionArray

import moveit_commander
from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
import moveit_msgs.msg

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import roslib

from std_srvs.srv import Empty
import kortex_driver.msg
import kortex_driver.srv
from kortex_driver.msg import Base_JointSpeeds, JointSpeed
from kortex_driver.msg import ActionNotification, ActionEvent
from kortex_driver.srv import ReadAction, ReadActionRequest, ExecuteAction, ExecuteActionRequest

import actionlib
from kortex_driver.srv import *
from kortex_driver.msg import *

from manipulator_control import jparse_cls 



class ExampleInitializeGazeboRobot(object):
    """Unpause Gazebo and home robot"""
    def __init__(self):
        # Initialize the node
        self.HOME_ACTION_IDENTIFIER = 2
        self.last_action_notif_type = None

        try:
            self.robot_name = rospy.get_param('~robot_name',"my_gen3")
            self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification, self.cb_action_topic)

            # Wait for the driver to be initialised
            while not rospy.has_param("/" + self.robot_name + "/is_initialized"):
                time.sleep(0.1)
                
            # Init the services
            read_action_full_name = '/' + self.robot_name + '/base/read_action'
            rospy.wait_for_service(read_action_full_name)
            self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

            execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
            rospy.wait_for_service(execute_action_full_name)
            self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)
        except rospy.ROSException as e:
            self.is_init_success = False
        else:
            self.is_init_success = True

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                return False
            else:
                time.sleep(0.01)

    def home_the_robot(self):
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        self.last_action_notif_type = None
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER

        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        # Execute the HOME action if we could read it
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot home...")
            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ExecuteAction")
                return False
            else:
                return self.wait_for_action_end_or_abort()



class ArmController:
    def __init__(self):
        rospy.init_node('arm_controller', anonymous=True)
        
        self.base_link = rospy.get_param('/base_link_name', 'base_link') #defaults are for Kinova gen3
        self.end_link = rospy.get_param('/end_link_name', 'end_effector_link')

        # Set parameters
        self.position_only = rospy.get_param('/position_only', False) #boolean to control only position versus full pose

        self.is_sim = rospy.get_param('~is_sim', True) #boolean to control if the robot is in simulation or not

        self.phi_gain_position = rospy.get_param('~phi_gain_position', 2) #gain for the phi term in the JParse method
        self.phi_gain_angular = rospy.get_param('~phi_gain_angular', 2) #gain for the phi term in the JParse method
        
        self.jparse_gamma = rospy.get_param('~jparse_gamma', 0.2) #gamma for the JParse method
        #choose the method to use
        self.method = rospy.get_param('~method', 'JParse') #options are JParse, JacobianPseudoInverse, JacobianDampedLeastSquares, JacobianProjection, JacobianDynamicallyConsistentInverse
        self.show_jparse_ellipses = rospy.get_param('~show_jparse_ellipses', False) #boolean to control if the jparse ellipses are shown or not
        self.use_space_mouse = rospy.get_param('~use_space_mouse', False) #boolean to control if the space mouse is used or not
        self.use_space_mouse_jparse = rospy.get_param('~use_space_mouse_jparse', False) #boolean to control if the space mouse is used or not

        #decide if moveit is used or not
        self.use_moveit_bool = rospy.get_param('~use_moveit', False) #boolean to control if moveit is used or not

        # Initialize variables
        self.joint_states = None
        self.target_pose = None
        self.jacobian_calculator = jparse_cls.JParse(base_link=self.base_link, end_link=self.end_link)

        #For visual servoing
        self.visual_servoing_bool = rospy.get_param('~visual_servoing', False) #boolean to control if visual servoing is used or not
        self.camera_frame_name = rospy.get_param('~camera_link_name', 'camera_link') #name of the camera frame
        self.servo_tag = rospy.get_param('~tag_name','tag_1') #name of the tag to use for visual servoing
        self.apriltag_topic = rospy.get_param('~apriltag_topic','/tag_detections')

        #get tf listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tfros = tf.TransformerROS()
        self.tf_timeout = rospy.Duration(1.0)
        

        if self.is_sim == True:
            #if we are in simulation, use the joint_states and target_pose topics
            joint_states_topic = '/my_gen3/joint_states'
        else:
            joint_states_topic = '/my_gen3/joint_states'

        rospy.Subscriber(joint_states_topic, JointState, self.joint_states_callback)

        #determine where target pose is coming from:
        if self.visual_servoing_bool == True:
            #visual servoing
            rospy.Subscriber(self.apriltag_topic, AprilTagDetectionArray, self.apriltag_pose_visual_servoing_callback)
        else:
            #message publishing desired pose
            rospy.Subscriber('/target_pose', PoseStamped, self.target_pose_callback)
           
        #if using spacemouse
        rospy.Subscriber('/robot_action', TwistStamped, self.space_mouse_callback)

        self.joint_vel_command_real = rospy.Publisher('/my_gen3/in/joint_velocity', Base_JointSpeeds, queue_size=10)     

        #For MoveIt! control
        moveit_commander.roscpp_initialize(sys.argv)
        self.moveit_commander = moveit_commander.RobotCommander()
        try:
            self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
            if self.is_gripper_present:
                gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
                self.gripper_joint_name = gripper_joint_names[0]
            else:
                self.gripper_joint_name = ""
            self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

            # Create the MoveItInterface necessary objects
            arm_group_name = "arm"
            self.robot = moveit_commander.RobotCommander("robot_description")
            self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
            self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
            self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)

            if self.is_gripper_present:
                gripper_group_name = "gripper"
                self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

            rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
        except Exception as e:
            print (e)
            self.is_init_success = False
        else:
            self.is_init_success = True



        #For Kinova velocity control
        self.joint_vel_command = rospy.Publisher('/my_gen3/in/joint_velocity', Base_JointSpeeds, queue_size=1)
        
        #publish relevant topics
        self.manip_measure_pub = rospy.Publisher('/manip_measure', Float64, queue_size=10)
        self.inverse_cond_number = rospy.Publisher('/inverse_cond_number', Float64, queue_size=10)
        #have certain messages to store raw error
        self.pose_error_pub = rospy.Publisher('/pose_error', PoseStamped, queue_size=10)
        self.position_error_pub = rospy.Publisher('/position_error', Vector3, queue_size=10)
        self.orientation_error_pub = rospy.Publisher('/orientation_error', Vector3, queue_size=10)
        #have certain messages to store control error
        self.pose_error_control_pub = rospy.Publisher('/pose_error_control', PoseStamped, queue_size=10)
        self.position_error_control_pub = rospy.Publisher('/position_error', Vector3, queue_size=10)
        self.orientation_control_error_pub = rospy.Publisher('/orientation_error', Vector3, queue_size=10)
        #publish the current end effector pose and target pose
        self.current_end_effector_pose_pub = rospy.Publisher('/current_end_effector_pose', PoseStamped, queue_size=10)
        self.current_target_pose_pub = rospy.Publisher('/current_target_pose', PoseStamped, queue_size=10)

        #For Kinova velocity control
        self.HOME_ACTION_IDENTIFIER = 2
        self.last_action_notif_type = None
        self.robot_name = rospy.get_param('~robot_name', "my_gen3")
        # Wait for the driver to initialize
        while not rospy.has_param("/" + self.robot_name + "/is_initialized"):
            time.sleep(0.1)
        read_action_full_name = '/' + self.robot_name + '/base/read_action'
        rospy.wait_for_service(read_action_full_name)
        self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

        execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
        rospy.wait_for_service(execute_action_full_name)
        self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)
        
        if self.is_sim == True:
            rospy.wait_for_service('/my_gen3/base/send_joint_speeds_command')
            try:
                self.srv_joint_speeds = rospy.ServiceProxy('/my_gen3/base/send_joint_speeds_command', kortex_driver.srv.SendJointSpeedsCommand)
            except:
                rospy.logerr("Failed to connect to joint speeds service")
                pass
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
        self.speed_req = kortex_driver.srv.SendJointSpeedsCommandRequest()

        joint_idx = []
        for jidx,jname in enumerate(self.joint_names):
            joint_speed= kortex_driver.msg.JointSpeed()
            joint_speed.joint_identifier= jidx
            joint_speed.value= 0.0
            self.speed_req.input.joint_speeds.append(joint_speed)
            joint_idx.append(jidx)
        #now zip the joint names and indices
        self.joint_idx_dict = dict(zip(self.joint_names, joint_idx))
        #Now start the control loop
        self.rate = rospy.Rate(20)  # 10 Hz
        
        # publisher of self.target_pose which is PoseStamped
        self.target_pose_pub = rospy.Publisher('/target_pose_servoing', PoseStamped, queue_size=10)

        # Start the control loop
        # hang out here for a bit to let the robot settle
        self.control_loop()

  
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
            
            

    def apriltag_pose_visual_servoing_callback(self,msg):
        """
        1. Looks up TF from the AprilTag (self.servo_tag) to the robot's base frame (self.base_link).
        2. Extracts the tag's Z-axis in base coordinates.
        3. Performs a dot-product with the base frame's X-axis to check if the tag is facing the robot.
        4. If yes, offsets the end-effector target pose 0.1m along the tag's Z-axis (in base frame).
        5. Stores the result in self.target_pose (PoseStamped).
        We use tf directly instead of the msg, but the msg serves as a good trigger
        """
        rospy.loginfo("Tag is detected")
        if self.target_pose is None:
            self.target_pose = PoseStamped()
            
        self.target_pose_pub.publish(self.target_pose)
        
        rospy.loginfo("Looking up transform from tag to base")
        try:
            # Lookup transform: from tag -> base
            tag_to_base: TransformStamped = self.tfBuffer.lookup_transform(
                source_frame= self.servo_tag,       # "base_link"
                target_frame=self.base_link,       # "tag_1"
                time=rospy.Time(0),               # most recent transform
                timeout=rospy.Duration(1.0)
            )
            rospy.loginfo(f"Transform from {self.servo_tag} to {self.base_link} found")
        except tf2_ros.LookupException as e:
            rospy.logerr(f"Transform not found: {e}")
            return
        except tf2_ros.ExtrapolationException as e:
            rospy.logerr(f"Transform error: {e}")
            return

        # Extract translation
        tx = tag_to_base.transform.translation.x
        ty = tag_to_base.transform.translation.y
        tz = tag_to_base.transform.translation.z

        # Extract quaternion and build a 4x4 transform matrix
        rx = tag_to_base.transform.rotation.x
        ry = tag_to_base.transform.rotation.y
        rz = tag_to_base.transform.rotation.z
        rw = tag_to_base.transform.rotation.w
        tag_quat = [rx, ry, rz, rw]
        tag_matrix = tf.transformations.quaternion_matrix(tag_quat)

        # The tag's Z-axis in its local frame is (0, 0, 1).
        # We transform that into the base frame using the 4x4 matrix.
        tag_z_axis_in_base = [
            tag_matrix[0, 2],
            tag_matrix[1, 2],
            tag_matrix[2, 2]
        ]

        # The base frame's X-axis is simply (1, 0, 0).
        # Dot product to check if the tag is facing the robot
        base_x_axis = [1.0, 0.0, 0.0]
        dot_product = (tag_z_axis_in_base[0] * base_x_axis[0] +
                       tag_z_axis_in_base[1] * base_x_axis[1] +
                       tag_z_axis_in_base[2] * base_x_axis[2])

        if dot_product  < 0:
            rospy.loginfo("Tag is facing the robot. Computing target pose...")

            # Offset 0.1 meters along the tag's Z-axis
            offset_distance = 0.4
            x_target = tx + offset_distance * tag_z_axis_in_base[0]
            y_target = ty + offset_distance * tag_z_axis_in_base[1]
            z_target = tz + offset_distance * tag_z_axis_in_base[2]

            # Fill self.target_pose
            self.target_pose.header.stamp = rospy.Time.now()
            self.target_pose.header.frame_id = self.base_link
            self.target_pose.pose.position.x = x_target
            self.target_pose.pose.position.y = y_target
            self.target_pose.pose.position.z = z_target
            
            # rotate the tag's orientation 
            # first make a rotation matrix from the quaternion
            angle = 2 * np.arccos(tag_quat[3])
            axis = np.array([tag_quat[0], tag_quat[1], tag_quat[2]])/np.sin(angle/2)
            tag_rotation_matrix = self.axis_angle_to_rotation_matrix(axis*angle)
        
            R_ee_to_tag = np.array([[-1, 0, 0],  
                                      [0, 1, 0],
                                      [0, 0, -1]])
            
            R_ee = np.dot(tag_rotation_matrix, R_ee_to_tag)
            
            # convert the rotation matrix to a quaternion
            q_ee = self.rotation_matrix_to_quaternion(R_ee)
            rw, rx, ry, rz = q_ee
            
            # Orientation: you might just copy the tag's orientation,
            # or set a specific orientation for your end effector.
            self.target_pose.pose.orientation.x = rx
            self.target_pose.pose.orientation.y = ry
            self.target_pose.pose.orientation.z = rz
            self.target_pose.pose.orientation.w = rw
            
            rospy.loginfo(f"Target pose set:\n{self.target_pose}")
        else:
            rospy.loginfo("Tag is NOT facing the robot. No action taken.")




    def target_pose_callback(self, msg):
        """
        Callback function for the target_pose topic. This function will be called whenever a new message is received
        on the target_pose topic. The message is a geometry_msgs/PoseStamped message, which contains the target pose
        """
        self.target_pose = msg

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
        
        if angle == 0:
            return np.eye(3)
        
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
        rospy.loginfo(f"Rotation matrix: {R}")
        if not (R.shape == (3, 3) and np.allclose(np.dot(R.T, R), np.eye(3)) and np.isclose(np.linalg.det(R), 1)):
            raise ValueError("Input matrix must be a valid rotation matrix.") 
        # Compute the quaternion using the method from the book
        q = np.zeros(4)
        q[0] = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
        q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
        q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
        q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])
        
        return q

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
        val = (np.trace(R) - 1) / 2
        angle = np.arccos(np.around(val, 4))
        
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
        rospy.loginfo(f"R_error: {R_error}")
        # Extract the axis-angle lie algebra vector from the rotation matrix
        angle_error = self.rotation_matrix_to_axis_angle(R_error)
        # Return the position and orientation error
        return position_error, angle_error


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


    def nominal_joint_pose_nullspace(self, q, Vmax = 5, Kp_value_set=1.0):
        """
        This function computes the nominal joint pose for the nullspace control of the Panda robot.
        - Vmax is the maximum joint error norm.
        """
        # Nominal joint pose for the Kinova robot
        if self.is_sim == True:
            q_nominal = np.array([0.00, -0.794, 0.00, -2.364, 0.00, 1.583, 0.785]) #joints panda_joint1 to panda_joint7
        else:
            q_nominal = np.array([0.00, 0.26, 3.14, -2.27, 0.00, 0.96, 1.57])
        # Compute the joint pose error
        q_error = q_nominal -q
        q_error_norm = np.linalg.norm(q_error)
        nominal_joint_motion = np.matrix(np.zeros(len(q))).T #if q_error_norm is less than 0.1, then return a zero vector
        if q_error_norm > 0.1:
            g = np.min([q_error_norm, Vmax])
            Kp = np.diag([Kp_value_set] * len(q))
            # Compute the nominal joint motion
            nominal_joint_motion = (g/q_error_norm) * Kp@q_error
            nominal_joint_motion = np.matrix(nominal_joint_motion).T
            
        return nominal_joint_motion


    
    def control_loop(self):
        while not rospy.is_shutdown():
            if self.joint_states and self.target_pose:
                # obtain the current joints
                q = []
                dq = []
                effort = []
                for joint_name in self.joint_names:
                    idx = self.joint_states.name.index(joint_name)
                    q.append(self.joint_states.position[idx])
                    dq.append(self.joint_states.velocity[idx])
                    effort.append(self.joint_states.effort[idx])  
                # Calculate the Method inv Jacobian
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

                
                manip_measure = self.jacobian_calculator.manipulability_measure(q)
                self.manip_measure_pub.publish(manip_measure)
                inverse_cond_number = self.jacobian_calculator.inverse_condition_number(q)
                self.inverse_cond_number.publish(inverse_cond_number)
                rospy.loginfo("Manipulability measure: %f", manip_measure)
                rospy.loginfo("Inverse condition number: %f", inverse_cond_number)


                #Calculate the delta_x (task space error)
                target_pose = self.target_pose
                current_pose = self.EndEffectorPose(q)

                current_end_effector_velocity = self.EndEffectorVelocity(q, dq) #full twist of the end effector
                position_error, angular_error = self.pose_error(current_pose, target_pose, control_pose_error=False)
                # log the pose error
                self.publish_pose_errors(position_error, angular_error, control_pose_error=False)
                if self.is_sim == False:
                    #real world limits
                    error_norm_max = 0.10
                else:
                    #simulation limits
                    error_norm_max = 0.30
                position_error, angular_error = self.pose_error(current_pose, target_pose, error_norm_max = error_norm_max, control_pose_error=True)                
                self.publish_pose_errors(position_error, angular_error, control_pose_error=True)
                
                #concatenate the position and angular error
                full_pose_error = np.matrix(np.concatenate((position_error, angular_error))).T
                #obtain nullspace control
                if self.is_sim == True:
                    nominal_motion_nullspace = self.nominal_joint_pose_nullspace(q=q, Vmax=5, Kp_value_set=2.0) 
                else:
                    nominal_motion_nullspace = self.nominal_joint_pose_nullspace(q=q, Vmax=5, Kp_value_set=0.0)
                # Calculate and command the joint velocities
                if self.position_only == True:
                    rospy.loginfo("Position only control")
                    position_error = np.matrix(position_error).T
                    if self.is_sim == True:
                        #use these gains only in simulation!
                        kp_gain = 1.0
                    else:
                        kp_gain = 1.0
                    Kp = np.diag([kp_gain, kp_gain, kp_gain])  # Proportional gain matrix

                    task_vector = Kp @ position_error 
                    joint_velocities = J_method @ task_vector + J_nullspace @ nominal_motion_nullspace
                else:
                    # realworld gains (tested)
                    kp_gain_position = 1.0
                    kp_gain_orientation = 1.0
                    if self.is_sim == True:
                        #use these gains only in simulation!
                        kp_gain_position = 1.0
                        kp_gain_orientation = 1.0
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
                
                joint_velocities_list = np.array(joint_velocities).flatten().tolist()

                joint_vel_dict = dict(zip(self.joint_names, joint_velocities_list)) #this dictionary combines the joint names and velocities
                # command the joint velocities
                if self.use_moveit_bool:
                    success = self.reach_joint_angles(joint_velocities=joint_velocities_list, tolerance=0.001, control_frequency=5) 
                    print("Joint velocities:", joint_velocities_list)
                    print("Success:", success)

                else:
                    self.command_joint_velocities(joint_vel_dict) #this commands the joint velocities

            rospy.loginfo("Control loop running")



    def reach_joint_angles(self, tolerance=0.01, control_frequency=20, 
                        joint_velocities=[0.0]*7,
                        velocity_scaling_factor=1, acceleration_scaling_factor=1):
        arm_group = self.arm_group
        success = True

        # Increase the effective time_step if the increments are too small.
        # For example, use a scaling factor or accumulate multiple increments.
        time_step = 1.0 / control_frequency  # base step

        # Get the current joint positions
        joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("Current joint positions: %s", joint_positions)

        # Adjust the target position based on commanded velocities.
        try:
            for i in range(len(joint_positions)):
                # Here, you might apply a velocity scaling factor or accumulate changes.
                joint_positions[i] += joint_velocities[i] * time_step * velocity_scaling_factor

            # Set the new goal for MoveIt!
            arm_group.set_joint_value_target(joint_positions)
            self.arm_group.set_goal_joint_tolerance(tolerance)
            
            # Plan the trajectory to this new goal
            (success_flag, trajectory_message, planning_time, error_code) = arm_group.plan()
            if not success_flag:
                rospy.logerr("Planning failed with error code: %s", error_code)
                return False

            # Execute the planned trajectory
            success &= arm_group.go(wait=True)
        except moveit_commander.MoveItCommanderException as e:
            rospy.logerr("Planning failed with exception: %s", str(e))
            success = False

        new_joint_positions = arm_group.get_current_joint_values()
        rospy.loginfo("New joint positions: %s", new_joint_positions)
        return success

    def reach_named_position(self, target, velocity_scaling_factor=1, acceleration_scaling_factor=1):
        arm_group = self.arm_group
        
        # Going to one of those targets
        rospy.loginfo("Going to named target " + target)
        # Set the target
        arm_group.set_named_target(target)

        # Plan the trajectory
        (success_flag, trajectory_message, planning_time, error_code) = arm_group.plan()
        # Execute the trajectory and block while it's not finished
        return arm_group.execute(trajectory_message, wait=True) #could use retimed_plan instead of trajectory_message

    def publish_joint_velocities(self, joint_velocities):
        """
        Publishes the joint velocities to the /joint_velocities topic.
        """
        joint_vel_msg = Float64MultiArray()
        joint_velocities = np.array(joint_velocities).flatten().tolist()
        joint_vel_msg.data = joint_velocities
        self.joint_vel_pub.publish(joint_vel_msg)


    def command_joint_velocities(self, joint_vel_dict):
        """
        This function commands the joint velocities to the robot using the appropriate ROS message type.
        """
        if self.is_sim == False:
            vel_msg = Base_JointSpeeds()
            # vel_msg.duration =0.01 #shouldn't need to set this
            for idx in range(len(self.joint_names)):
                
                joint_speed = JointSpeed()
                joint_speed.joint_identifier = idx
                joint_speed.value = joint_vel_dict[self.joint_names[idx]]
                vel_msg.joint_speeds.append(joint_speed)
                
            rospy.loginfo("SENT Joint velocities: %s", vel_msg.joint_speeds)
            
            self.joint_vel_command_real.publish(vel_msg)
        else:
            for joint_speed in self.speed_req.input.joint_speeds:
                for name in self.joint_idx_dict: 
                    if joint_speed.joint_identifier == self.joint_idx_dict[name]:             
                        joint_speed.value= joint_vel_dict[name] #this is the joint velocity

                #the joint_speed.joint_identifier is the joint index, which is used to find the joint velocity name from self.joint_idx_dict, which is then used to find the joint velocity value from joint_vel_dict
                self.srv_joint_speeds.call(self.speed_req)

if __name__ == '__main__':
    try:
        controller = ArmController()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Create a dictionary with zero velocities for all joints
        zero_vel_dict = {joint_name: 0.0 for joint_name in controller.joint_names}
        # Send zero velocities to stop the robot safely
        controller.command_joint_velocities(zero_vel_dict)
        rospy.loginfo("Program interrupted: Sent zero joint velocities for safe shutdown")