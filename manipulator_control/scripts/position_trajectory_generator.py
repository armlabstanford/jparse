#!/usr/bin/env python3

import rospy
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseStamped

class OvalTrajectory:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('oval_trajectory_generator', anonymous=True)

        # RViz Marker publisher
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

        # Pose publisher
        self.pose_pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)

        # Parameters for the oval trajectory
        self.center = rospy.get_param('~center', [0.0, 0.0, 0.0])  # Center of the oval [x, y, z]
        if type(self.center) is str:
            self.center = [float(x) for x in self.center.strip('[]').split(',')] # Convert string to list of floats

        self.major_axis = float(rospy.get_param('~major_axis', "1.0"))  # Length of the major axis
        self.minor_axis = float(rospy.get_param('~minor_axis', "0.5"))  # Length of the minor axis
        self.pitch_axis = float(rospy.get_param('~pitch_axis', "0.0"))  # Length of the pitch axis
        self.plane = rospy.get_param('~plane', "xy")  # Plane of the oval ('xy', 'yz', or 'xz')
        self.frequency = float(rospy.get_param('~frequency', "0.2"))  # Frequency of the trajectory (Hz)
        self.base_frame = rospy.get_param('~base_frame', "base_link")  # Base frame for the trajectory

        # Time tracking
        self.start_time = rospy.Time.now()

        # Start the trajectory generation loop
        self.trajectory_loop()

    def generate_position(self, time_elapsed):
        """Generate the 3D position of the point on the oval."""
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

    def publish_marker(self, position):
        """Publish the current goal position as an RViz marker."""
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "oval_trajectory"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.marker_pub.publish(marker)

    def publish_pose(self, position):
        """Publish the current goal position as a geometry_msgs/Pose message."""
        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.pose_pub.publish(pose)

    def trajectory_loop(self):
        """Main loop for trajectory generation and marker/pose publishing."""
        rate = rospy.Rate(50)  # 50 Hz control loop

        while not rospy.is_shutdown():
            # Calculate elapsed time
            elapsed_time = (rospy.Time.now() - self.start_time).to_sec()

            # Generate current position on the oval
            position = self.generate_position(elapsed_time)

            # Publish the marker and pose
            self.publish_marker(position)
            self.publish_pose(position)

            # Sleep to maintain loop rate
            rate.sleep()


if __name__ == '__main__':
    try:
        OvalTrajectory()
    except rospy.ROSInterruptException:
        rospy.loginfo("Oval trajectory generator node terminated.")