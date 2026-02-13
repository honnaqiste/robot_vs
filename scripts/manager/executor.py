#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PoseStamped
from robot_vs.msg import RobotCommand

class Executor:
    def __init__(self, namespace):
        self.ns = namespace
        
        # 发布导航目标点
        self.goal_pub = rospy.Publisher(
            "/{}/move_base_simple/goal".format(self.ns),
            PoseStamped,
            queue_size=10
        )
        
        # 发布自定义行为指令
        self.command_pub = rospy.Publisher(
            "/{}/robot_command".format(self.ns),
            RobotCommand,
            queue_size=10
        )
        
        rospy.loginfo("[{}] 执行模块初始化完成".format(self.ns))

    def publish_nav_goal(self, goal):
        """发布导航目标点到move_base"""
        
        self.goal_pub.publish(goal)
        # rospy.loginfo("[{}] 发布导航目标: ({}, {})".format(self.ns,goal.pose.position.x,goal.pose.position.y))

    def publish_robot_command(self, cmd):
        """发布自定义行为指令"""

        
        self.command_pub.publish(cmd)
        # rospy.loginfo("[{}] 发布行为指令: 攻击={}, 模式={}".format(self.ns,cmd.attack,cmd.mode))