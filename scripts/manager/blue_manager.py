#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from robot_vs.msg import RobotCommand

class TeamManagerBlue:
    def __init__(self):
        rospy.init_node('team_manager_blue')
        
        rospy.Subscriber('/robot_blue/odom', Odometry, self.odom_cb)
        self.pub = rospy.Publisher('/robot_blue/cmd_vel', Twist, queue_size=10)
        
        rospy.loginfo("【蓝方Manager】启动完成")

    def odom_cb(self, msg):
        pass

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            vel = Twist()
            vel.linear.x = 0.3
            vel.angular.z = 0.0
            self.pub.publish(vel)
            rate.sleep()

if __name__ == '__main__':
    try:
        TeamManagerBlue().run()
    except rospy.ROSInterruptException:
        pass