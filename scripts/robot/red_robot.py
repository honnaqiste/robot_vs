from command_parser import CommandParser
import rospy

class RedRobot:
    def __init__(self):
        rospy.init_node("red_robot")
        self.ns = "robot_red"  # 红方命名空间
        
        self.parser = CommandParser(self.ns)

        self.rate = rospy.Rate(10)  # 频率
        rospy.loginfo("红方TeamManager启动完成")

    def run(self):
        while not rospy.is_shutdown():

            self.rate.sleep()



if __name__ == '__main__':
    try:
        robot = RedRobot()
        robot.run()
    except rospy.ROSInterruptException:
        pass