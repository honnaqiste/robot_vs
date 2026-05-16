# robot_vs

本仓库实现了多机器人红蓝对抗系统，支持 **仿真环境 / 真实机器人 / LLM 决策 / 分层 MAS 多智能体决策** 等功能。

系统整体采用 **Manager + Car Agent + Skill** 三层架构，并新增了 **Referee 裁判系统**、**RViz 可视化系统** 与 **MAS 分层大模型规划模块**：

- **Manager 层**（`scripts/manager/`）：感知全局战场状态，调用规则决策 / LLM / MAS 服务生成战术任务，并通过 `TaskCommand` 向小车下发任务。
- **Car Agent 层**（`scripts/car/`）：每辆小车运行一个独立的 `car_node.py`，接收任务并驱动技能执行，同时周期性发布 `RobotState`。
- **Skill 系统**（`scripts/car/skills/`）：实现 GOTO / STOP / ATTACK / ROTATE 等原子动作。
- **Referee 裁判系统**（`scripts/manager/referee_node.py`）：负责血量、弹药、开火命中、视野可见性和宏观战况统计。
- **MAS 系统**（`scripts/MAS/`）：提供 LeaderAgent + CarAgent 分层多智能体 LLM 决策服务。

详细架构说明与数据流图请参阅 → **[技术原理文档](TECHNICAL.md)**

---

## 演示 / Demo

> 🎬 演示图/视频即将更新，敬请期待……

---

## 功能特性

- 支持 **Gazebo 仿真** 与 **真实机器人** 两套运行环境
- 支持 **红方 / 蓝方 2v2 多机器人对抗**
- 支持经典 **Manager 决策架构**
- 支持 **LLM 大模型战术规划**
- 支持 **MAS 分层多智能体决策**
  - LeaderAgent：低频战略规划
  - CarAgent：高频单车战术执行
  - STM / LTM：短期记忆与长期记忆
- 支持 `GOTO / STOP / ATTACK / ROTATE` 四类任务动作
- 支持裁判系统：
  - 血量统计
  - 弹药消耗
  - 开火命中判定
  - 视野范围与遮挡判断
  - 红蓝双方宏观状态发布
- 支持 RViz 可视化：
  - 血条
  - 弹药
  - 底盘阵营颜色
  - 弹道区域
  - move_base 路径
- 基于 **命名空间 + TF 前缀** 实现多机器人话题隔离
- 仿真与真机话题结构尽量保持一致，便于算法迁移

---

## 快速开始

详细环境搭建与依赖安装请参考 → **[环境配置文档](INSTALL.md)**

```bash
# 1. 克隆项目到 ROS 工作空间
cd ~/catkin_ws/src
git clone https://github.com/Xqrion/robot_vs.git

# 2. 编译
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```
## 仿真运行
### 启动 2v2 红蓝对抗仿真
```Bash
roslaunch robot_vs simulation/2v2vs_simulation.launch
```
该启动文件会自动拉起：

- Gazebo 多机器人仿真
- map_server
- AMCL
- move_base
- Referee 裁判节点
- RViz 可视化
- Manager
- Car Agent

## 经典 Manager + LLM 运行
### 1. 启动 LLM 服务
```Bash
bash config/AI/start_llm_services.sh
```
默认端口：

| 阵营 | 端口 |
|------|------|
| red |	8001 |
| blue | 8002 |

LLM 配置文件：

```text
config/AI/llm_config.yaml
```
支持模型配置：

- Kimi
- GLM
- Doubao
- DeepSeek
- Qwen
- 其他 OpenAI-compatible API
### 2. 启动 Manager
```Bash
roslaunch robot_vs manager/managers.launch
```
Manager 配置文件：
```text
config/manager/red_manager.yaml
config/manager/blue_manager.yaml
```
## MAS 分层多智能体运行
### MAS 是当前工程新增的高级决策模块，采用：
```text
LeaderAgent 低频战略规划
        ↓
CarAgent 高频单车决策
        ↓
TaskCommand 下发到机器人
```
### 启动 MAS 服务
```Bash
bash config/AI/start_mas_services.sh
```
MAS 默认同样监听：

| 阵营 | 端口 |
|------|------|
| red |	8001 |
| blue | 8002 |
MAS 配置文件：
```text
scripts/MAS/configs/models.yaml
scripts/MAS/configs/prompts_3.2a.yaml
```
## 真机运行
### 主控机启动地图、裁判与 Manager
```Bash
roslaunch robot_vs real_world/master/2v2vs_real.launch
```
### 单台机器人启动
```Bash
roslaunch robot_vs real_world/cars/robot_bringup.launch robot_namespace:=robot_red
```
可选命名空间：
```text
robot_red
robot_red2
robot_blue
robot_blue2
```
## 常用工具
### 停止所有机器人
```Bash
rosrun robot_vs stop_all_robots.py
```
该工具会：
- 取消 move_base 目标
- 发布零速度
- 可选清空 costmap
## 核心话题
| 话题 | 消息 | 说明 |
|------|------|------|
|/<ns>/task_cmd	|TaskCommand	|Manager 向小车下发任务|
|/<ns>/robot_state|	RobotState|	小车反馈自身状态|
|/<ns>/fire_event	|FireEvent	|小车开火事件|
|/referee/macro_state|	BattleMacroState	|裁判发布全局宏观状态|
|/red_manager/enemy_state|	VisibleEnemies	|红方可见敌人|
|/blue_manager/enemy_state|	VisibleEnemies	|蓝方可见敌人|


## 文档索引

| 文档 | 内容 |
|------|------|
| [环境配置](INSTALL.md) | 虚拟机搭建、ROS 安装、项目部署全流程 |
| [技术原理](TECHNICAL.md) | 系统架构、Manager/Car/Skill 详解、ROS 消息流、数据流图 |

---

## 项目状态

| 模块 | 状态 |
|------|------|
| 仿真环境（Gazebo + Rviz） | ✅ 已完成 |
| 红蓝阵营 Manager 框架 | ✅ 已完成 |
| 真机局域网下通信测试 | ✅ 已完成 |
| Car Agent + Skill 系统 | ✅ 已完成 |
| GOTO / STOP / ATTACK / ROTATE 技能 | ✅ 已完成 |
| 裁判系统 | ✅ 已完成 |
| RViz 血条 / 弹药 / 弹道可视化 | ✅ 已完成 |
| 单模型 LLM 接入 | ✅ 已完成 |
| MAS 分层多智能体系统	| ✅ 已完成 |
|真机局域网通信	|✅ 已完成|
|真机 2v2 联调	|🚧 进行中|
|战术 Prompt 优化	|🚧 进行中|

## 说明
本项目主要用于：
- 多机器人红蓝对抗实验
- ROS 多机器人系统开发
- LLM / MAS 战术规划研究
- 仿真到真机迁移验证
