#!/usr/bin/env python3

import rospy
import math
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler, quaternion_matrix

class SE3Trajectory:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('se3_trajectory_generator', anonymous=True)

        # RViz Marker publisher
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

        # Pose publisher
        self.pose_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)

        # Start here publisher
        self.start_here_pub = rospy.Publisher('/start_here', Header, queue_size=10)

        # Parameters for position trajectory
        self.center = rospy.get_param('~center', [0.0, 0.0, 0.0])  # Center of the oval [x, y, z]
        if type(self.center) is str:
            self.center = [float(x) for x in self.center.strip('[]').split(',')] # Convert string to list of floats
        
        self.robot = rospy.get_param('~robot', "panda")  # Robot type ('panda' or 'xarm' or 'kinova')

        self.key_points_only_bool = rospy.get_param('~key_points_only_bool', "False")  # Boolean to determine if only key points should be published
        self.use_rotation = rospy.get_param('~use_rotation', "True")  # Boolean to determine if rotation should be used

        self.major_axis = float(rospy.get_param('~major_axis', "1.0"))  # Length of the major axis
        self.minor_axis = float(rospy.get_param('~minor_axis', "0.5"))  # Length of the minor axis
        self.pitch_axis = float(rospy.get_param('~pitch_axis', "0.0"))  # Length of the pitch axis


        self.plane = rospy.get_param('~plane', "xy")  # Plane of the oval ('xy', 'yz', or 'xz')
        self.frequency = float(rospy.get_param('~frequency', "0.2"))  # Frequency of the trajectory (Hz)

        # Parameters for orientation trajectory
        self.orientation_major_axis = float(rospy.get_param('~orientation_major_axis', "0.3"))
        self.orientation_minor_axis = float(rospy.get_param('~orientation_minor_axis', "0.1"))
        self.orientation_frequency = float(rospy.get_param('~orientation_frequency', "0.1"))  # Frequency of orientation change

        #Get the base frame for the trajectory
        self.base_frame = rospy.get_param('~base_frame', "base_link")  # Base frame for the trajectory

        #set scale for marker arrow length
        self.arrow_scale = 0.25

        # Time tracking
        self.start_time = rospy.Time.now()

        # Start the trajectory generation loop
        self.trajectory_loop()

    def discrete_positions_around_ellipse(self, time_elapsed):
        # Period (time spent at each critical point)
        period = 1 / self.frequency
        # Determine which critical point to select based on time_elapsed
        critical_point_index = int((time_elapsed % (4 * period)) // period)

        if self.plane == 'xy':
            if critical_point_index == 0:  # Positive major axis
                x = self.center[0] + self.major_axis
                y = self.center[1]
                z = self.center[2]
            elif critical_point_index == 1:  # Positive minor axis
                x = self.center[0]
                y = self.center[1] + self.minor_axis
                z = self.center[2]
            elif critical_point_index == 2:  # Negative major axis
                x = self.center[0] - self.major_axis
                y = self.center[1]
                z = self.center[2]
            elif critical_point_index == 3:  # Negative minor axis
                x = self.center[0]
                y = self.center[1] - self.minor_axis
                z = self.center[2]

        elif self.plane == 'yz':
            if critical_point_index == 0:  # Positive major axis
                x = self.center[0]
                y = self.center[1] + self.major_axis
                z = self.center[2]
            elif critical_point_index == 1:  # Positive minor axis
                x = self.center[0]
                y = self.center[1]
                z = self.center[2] + self.minor_axis
            elif critical_point_index == 2:  # Negative major axis
                x = self.center[0]
                y = self.center[1] - self.major_axis
                z = self.center[2]
            elif critical_point_index == 3:  # Negative minor axis
                x = self.center[0]
                y = self.center[1]
                z = self.center[2] - self.minor_axis

        elif self.plane == 'xz':
            if critical_point_index == 0:  # Positive major axis
                x = self.center[0] + self.major_axis
                y = self.center[1]
                z = self.center[2]
            elif critical_point_index == 1:  # Positive minor axis
                x = self.center[0]
                y = self.center[1]
                z = self.center[2] + self.minor_axis
            elif critical_point_index == 2:  # Negative major axis
                x = self.center[0] - self.major_axis
                y = self.center[1]
                z = self.center[2]
            elif critical_point_index == 3:  # Negative minor axis
                x = self.center[0]
                y = self.center[1]
                z = self.center[2] - self.minor_axis

        else:
            rospy.logwarn("Invalid plane specified. Defaulting to 'xy'.")
            if critical_point_index == 0:  # Positive major axis
                x = self.center[0] + self.major_axis
                y = self.center[1]
                z = self.center[2]
            elif critical_point_index == 1:  # Positive minor axis
                x = self.center[0]
                y = self.center[1] + self.minor_axis
                z = self.center[2]
            elif critical_point_index == 2:  # Negative major axis
                x = self.center[0] - self.major_axis
                y = self.center[1]
                z = self.center[2]
            elif critical_point_index == 3:  # Negative minor axis
                x = self.center[0]
                y = self.center[1]
                z = self.center[2] - self.minor_axis

        return x, y, z

    def generate_position(self, time_elapsed):
        """Generate the 3D position of the point on the oval."""
        if self.key_points_only_bool:
            x, y, z = self.discrete_positions_around_ellipse(time_elapsed)

        else:
            angle = 2 * math.pi * self.frequency * time_elapsed

            if self.plane == 'xy':
                x = self.center[0] + self.major_axis * math.cos(angle)
                y = self.center[1] + self.minor_axis * math.sin(angle)
                z = self.center[2] + self.pitch_axis * math.cos(angle)
            elif self.plane == 'yz':
                x = self.center[0]
                y = self.center[1] + self.major_axis * math.cos(angle)
                z = self.center[2] + self.minor_axis * math.sin(angle)
            elif self.plane == 'xz':
                x = self.center[0] + self.major_axis * math.cos(angle)
                y = self.center[1]
                z = self.center[2] + self.minor_axis * math.sin(angle)
            else:
                rospy.logwarn("Invalid plane specified. Defaulting to 'xy'.")
                x = self.center[0] + self.major_axis * math.cos(angle)
                y = self.center[1] + self.minor_axis * math.sin(angle)
                z = self.center[2]



        return x, y, z

    def generate_orientation(self, time_elapsed):
        """
        Generate the orientation as a quaternion tracing an oval.
        this function returns a quaternion representing the orientation of the goal pose.
        the type of the returned value is a list of 4 floats representing the quaternion in the order [x, y, z, w].
        """
        angle = 2 * math.pi * self.orientation_frequency * time_elapsed

        if self.use_rotation:
            roll = np.pi #keep roll constant
            pitch =  self.orientation_major_axis * math.cos(angle)   # Oval in pitch
            yaw = self.orientation_minor_axis * math.sin(angle)  # Oval in yaw
        else:
            if self.robot == "panda" or self.robot == "xarm":
                roll = np.pi
                pitch = 0 #self.orientation_major_axis
                yaw = 0 #self.orientation_minor_axis
            elif self.robot == "kinova":
                roll = np.pi/2
                pitch = 0
                yaw = np.pi/2          

        # Generate quaternion from roll, pitch, yaw
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        return quaternion

    def publish_marker(self, position, orientation):
        """Publish the current goal position and orientation as RViz markers representing axes."""
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_matrix(orientation)

        # Define axis vectors (1 unit length)
        x_axis = rotation_matrix[:3, 0]  # Red
        y_axis = rotation_matrix[:3, 1]  # Green
        z_axis = rotation_matrix[:3, 2]  # Blue

        # Publish arrows for each axis
        self.publish_arrow(position, x_axis, [1.0, 0.0, 0.0, 1.0], 0)  # Red
        self.publish_arrow(position, y_axis, [0.0, 1.0, 0.0, 1.0], 1)  # Green
        self.publish_arrow(position, z_axis, [0.0, 0.0, 1.0, 1.0], 2)  # Blue

    def publish_arrow(self, position, axis_vector, color, marker_id):
        """Publish an arrow marker for a given axis."""
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "se3_trajectory"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Arrow start and end points
        marker.points = []
        start_point = Pose().position
        start_point.x = position[0]
        start_point.y = position[1]
        start_point.z = position[2]
        end_point = Pose().position
        end_point.x = position[0] + self.arrow_scale*axis_vector[0]
        end_point.y = position[1] + self.arrow_scale*axis_vector[1]
        end_point.z = position[2] + self.arrow_scale*axis_vector[2]
        marker.points.append(start_point)
        marker.points.append(end_point)

        # Arrow properties
        marker.scale.x = 0.02  # Shaft diameter
        marker.scale.y = 0.05  # Head diameter
        marker.scale.z = 0.05  # Head length
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        self.marker_pub.publish(marker)

    def publish_pose(self, position, orientation):
        """Publish the current goal position and orientation as a geometry_msgs/Pose message."""
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]

        self.pose_pub.publish(pose)

    def trajectory_loop(self):
        """Main loop for trajectory generation and marker/pose publishing."""
        rate = rospy.Rate(50)  # 50 Hz control loop

        # Wait for subscribers to connect
        rospy.sleep(1.0)

        # Send start_here topic before main loop
        start_msg = Header()
        start_msg.stamp = rospy.Time.now()
        start_msg.frame_id = self.base_frame
        self.start_here_pub.publish(start_msg)
        rospy.loginfo("Published start_here signal")

        # Give time for rosbag to record
        rospy.sleep(0.5)

        while not rospy.is_shutdown():
            # Calculate elapsed time
            elapsed_time = (rospy.Time.now() - self.start_time).to_sec()

            # Generate position and orientation
            position = self.generate_position(elapsed_time)
            orientation = self.generate_orientation(elapsed_time)

            # Publish the marker and pose
            self.publish_marker(position, orientation)
            self.publish_pose(position, orientation)

            # Sleep to maintain loop rate
            rate.sleep()


if __name__ == '__main__':
    try:
        SE3Trajectory()
    except rospy.ROSInterruptException:
        rospy.loginfo("SE(3) trajectory generator node terminated.")