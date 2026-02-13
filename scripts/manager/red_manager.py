#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from perception import Perception
from decision import DecisionEngine
from executor import Executor

class RedTeamManager:
    def __init__(self):
        # 初始化节点，指定命名空间
        rospy.init_node("red_team_manager")
        self.ns = "robot_red"  # 红方命名空间
        
        # 初始化三大模块（可扩展：新增模块只需加初始化）
        self.perception = Perception(self.ns)  # 感知
        self.decision = DecisionEngine(self.ns)# 决策
        self.executor = Executor(self.ns)      # 执行

        self.rate = rospy.Rate(10)  # 决策频率
        rospy.loginfo("红方TeamManager启动完成")

    def run(self):
        """主循环：感知→决策→执行"""
        while not rospy.is_shutdown():
            # 1. 感知：获取小车的位置/图像数据
            pose = self.perception.get_current_pose()

            image = self.perception.get_current_image()

            # 2. 决策：根据感知数据生成指令
            nav_goal, robot_cmd = self.decision.make_decision(pose, image)

            # 3. 执行：发布指令给小车
            if nav_goal:
                self.executor.publish_nav_goal(nav_goal)
            if robot_cmd:
                self.executor.publish_robot_command(robot_cmd)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        manager = RedTeamManager()
        manager.run()
    except rospy.ROSInterruptException:
        pass