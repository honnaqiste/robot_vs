#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PoseStamped
from robot_vs.msg import RobotCommand

class DecisionEngine:
    def __init__(self, namespace):
        self.ns = namespace
        # 可扩展：加载决策参数（如巡逻路径、攻击阈值）
        self.patrol_points = [(0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]  # 巡逻点
        self.current_patrol_idx = 0

    def make_decision(self, pose, image):
        """核心决策逻辑（可扩展：新增强化学习/规则逻辑）"""
        if pose is None:
            return None, None

        # 示例：巡逻逻辑（可扩展：加攻击判定、避障逻辑）
        current_x = pose.position.x
        current_y = pose.position.y
        target_x, target_y = self.patrol_points[self.current_patrol_idx]

        # 判断是否到达当前巡逻点
        if abs(current_x - target_x) < 0.05 and abs(current_y - target_y) < 0.05:
            self.current_patrol_idx = (self.current_patrol_idx + 1) % len(self.patrol_points)
            target_x, target_y = self.patrol_points[self.current_patrol_idx]
            rospy.loginfo("Arrived one target")

        # 构造导航目标
        nav_goal = PoseStamped()
        nav_goal.header.frame_id = "map"
        nav_goal.pose.position.x = target_x
        nav_goal.pose.position.y = target_y
        nav_goal.pose.orientation.w = 1.0

        # 构造行为指令（巡逻模式，不攻击）
        robot_cmd = RobotCommand()
        robot_cmd.mode = 1
        robot_cmd.attack = False
        robot_cmd.goal_x = target_x
        robot_cmd.goal_y = target_y

        return nav_goal, robot_cmd