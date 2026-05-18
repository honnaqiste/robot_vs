#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import json
import math
import os
import threading
import time
from collections import OrderedDict

import rospy
from robot_vs.msg import BattleMacroState
from robot_vs.msg import EnemyInfo
from robot_vs.msg import FireEvent
from robot_vs.msg import GameState
from robot_vs.msg import MatchRecord
from robot_vs.msg import RobotMatchStat
from robot_vs.msg import RobotState
from robot_vs.msg import TaskCommand
from robot_vs.msg import TeamMacroState
from robot_vs.msg import TeamMatchStat
from robot_vs.msg import VisibleEnemies
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String


class RefereeNode(object):
    """全局唯一裁判节点。

    功能：
    1) 动态发现并订阅 /<ns>/robot_state 与 /<ns>/fire_event
    2) 维护全局状态（位姿、阵营、HP、生死）
    3) 处理开火命中判定并扣血
    4) 周期发布双方可见敌人列表
    """

    def __init__(self):
        self.loop_hz = float(rospy.get_param("~loop_hz", 10.0))
        self.discover_hz = float(rospy.get_param("~discover_hz", 1.0))

        self.default_hp = int(rospy.get_param("~default_hp", 100))
        self.default_ammo = float(rospy.get_param("~default_ammo", 50.0))
        self.fire_range = float(rospy.get_param("~fire_range", 5.0))
        self.hit_width = float(rospy.get_param("~hit_width", 0.5))
        self.occlusion_width = float(rospy.get_param("~occlusion_width", self.hit_width))
        self.fire_damage = int(rospy.get_param("~fire_damage", 20))
        self.vision_range = float(rospy.get_param("~vision_range", 4.0))

        self.fov_deg = float(rospy.get_param("~fov_deg", 120.0))
        self.fov_rad = math.radians(self.fov_deg)
        self.map_topic = str(rospy.get_param("~map_topic", "/map"))
        self.occ_threshold = int(rospy.get_param("~occ_threshold", 50))  # 0~100, >=阈值视为障碍
        self.block_unknown = bool(rospy.get_param("~block_unknown", True))  # -1 unknown 是否当障碍

        # ====== 比赛管理参数 ======
        self.time_limit = float(rospy.get_param("~time_limit", 120.0))
        self.experiment = str(rospy.get_param("~experiment", ""))
        logs_dir_param = str(rospy.get_param("~logs_dir", ""))
        if logs_dir_param:
            self.logs_dir = logs_dir_param
        else:
            self.logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
        self.logs_dir = os.path.abspath(self.logs_dir)
        # 如果指定了实验名，在 logs 下建子目录
        if self.experiment:
            self.logs_dir = os.path.join(self.logs_dir, self.experiment)

        self._map_info = None
        self._map_data = None

        self._map_sub = rospy.Subscriber(self.map_topic, OccupancyGrid, self._on_map, queue_size=1)

        self._lock = threading.RLock()

        # dict[ns] = {"team", "x", "y", "yaw", "hp", "alive", "ammo"}
        self.global_states = {}

        self._robot_state_subs = {}
        self._fire_event_subs = {}

        # ====== 比赛生命周期 ======
        self._match_status = "IDLE"       # IDLE / PLAYING / FINISHED
        self._match_id = 0
        self._match_start_wall = 0.0       # time.time() when match started
        self._match_start_str = ""
        self._match_end_str = ""
        self._match_winner = ""
        self._match_reason = ""
        self._match_ended = False          # 防止重复结束

        # 每辆小车的对局统计
        # dict[ns] = {"kills":0, "deaths":0, "shots_fired":0, "hits_landed":0,
        #             "hits_taken":0, "damage_dealt":0.0, "damage_taken":0.0,
        #             "death_time":None}
        self._match_stats = {}

        # 叙事时间线：实时追加写入文件
        self._narrative = []
        self._narrative_path = None
        self._team_log_paths = {"red": None, "blue": None}
        self._narrative_sub = rospy.Subscriber(
            "/game/narrative", String, self._on_narrative_event, queue_size=200
        )

        # ====== 停车发布器缓存 ======
        # dict[ns] -> Publisher 用于比赛结束时发 STOP
        self._stop_pubs = {}

        # ====== 发布器 ======
        self.red_enemy_pub = rospy.Publisher(
            "/red_manager/enemy_state", VisibleEnemies, queue_size=10
        )
        self.blue_enemy_pub = rospy.Publisher(
            "/blue_manager/enemy_state", VisibleEnemies, queue_size=10
        )
        self.macro_state_pub = rospy.Publisher(
            "/referee/macro_state", BattleMacroState, queue_size=10
        )
        self.game_state_pub = rospy.Publisher(
            "/game/state", GameState, queue_size=10
        )
        self.game_result_pub = rospy.Publisher(
            "/game/result", MatchRecord, queue_size=10
        )

        # ====== 比赛控制订阅 ======
        self._game_cmd_sub = rospy.Subscriber(
            "/game/command", String, self._on_game_command, queue_size=10
        )

        # 确保日志目录存在
        try:
            if not os.path.exists(self.logs_dir):
                os.makedirs(self.logs_dir)
        except Exception as exc:
            rospy.logwarn("[referee] cannot create logs dir %s: %s", self.logs_dir, exc)

        # 初始化叙事文件路径（比赛开始时创建）
        self._init_narrative_path()

        rospy.loginfo(
            "RefereeNode initialized: loop_hz=%.1f discover_hz=%.1f "
            "fire_range=%.2f hit_width=%.2f fire_damage=%d vision_range=%.2f "
            "time_limit=%.1f logs_dir=%s",
            self.loop_hz,
            self.discover_hz,
            self.fire_range,
            self.hit_width,
            self.fire_damage,
            self.vision_range,
            self.time_limit,
            self.logs_dir,
        )

    def _init_narrative_path(self):
        """设置叙事文件路径（比赛 ID 确定后调用）。"""
        if self._match_id <= 0:
            return
        self._narrative_path = os.path.join(
            self.logs_dir, "match_%03d_narrative.jsonl" % self._match_id
        )
        self._team_log_paths = {
            "red": os.path.join(self.logs_dir, "match_%03d_red_log.jsonl" % self._match_id),
            "blue": os.path.join(self.logs_dir, "match_%03d_blue_log.jsonl" % self._match_id),
        }

    def _clear_log_files(self):
        """清空本场比赛的日志文件（若存在则删除重建）。"""
        for path in [self._narrative_path] + list(self._team_log_paths.values()):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as exc:
                    rospy.logwarn("[referee] failed to remove old log %s: %s", path, exc)

    @staticmethod
    def _normalize_ns(ns):
        return str(ns).strip().strip("/")

    @staticmethod
    def _parse_ns_from_topic(topic, suffix):
        if not topic or not topic.startswith("/"):
            return None
        if not topic.endswith(suffix):
            return None
        ns = topic[1 : -len(suffix)]
        ns = ns.strip("/")
        return ns if ns else None

    @staticmethod
    def _detect_team(ns):
        value = str(ns).lower()
        if "red" in value:
            return "red"
        if "blue" in value:
            return "blue"
        return "unknown"

    @staticmethod
    def _decode_team_code(team_code):
        """把 RobotState.team 的数值编码转为字符串阵营。"""
        try:
            code = int(team_code)
        except (TypeError, ValueError):
            return "unknown"

        # 约定来自 car 配置：0=red, 1=blue
        if code == 0:
            return "red"
        if code == 1:
            return "blue"
        return "unknown"

    @staticmethod
    def _quaternion_to_yaw(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _ensure_robot_record(self, ns):
        ns = self._normalize_ns(ns)
        if not ns:
            return None

        record = self.global_states.get(ns)
        if record is not None:
            return record

        record = {
            "team": self._detect_team(ns),
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "hp": int(self.default_hp),
            "ammo": float(self.default_ammo),
            "alive": True,
        }
        self.global_states[ns] = record
        rospy.loginfo("[referee] tracking robot: ns=%s team=%s", ns, record["team"])
        return record

    def _discover_and_subscribe(self):
        try:
            topics = rospy.get_published_topics()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "get_published_topics failed: %s", exc)
            return

        for topic, msg_type in topics:
            if topic.endswith("/robot_state") and msg_type == "robot_vs/RobotState":
                ns = self._parse_ns_from_topic(topic, "/robot_state")
                if not ns:
                    continue
                with self._lock:
                    self._ensure_robot_record(ns)
                    if ns not in self._robot_state_subs:
                        self._robot_state_subs[ns] = rospy.Subscriber(
                            topic,
                            RobotState,
                            self._on_robot_state,
                            callback_args=ns,
                            queue_size=20,
                        )
                        rospy.loginfo("[referee] subscribed robot_state: %s", topic)

            if topic.endswith("/fire_event") and msg_type == "robot_vs/FireEvent":
                ns = self._parse_ns_from_topic(topic, "/fire_event")
                if not ns:
                    continue
                with self._lock:
                    self._ensure_robot_record(ns)
                    if ns not in self._fire_event_subs:
                        self._fire_event_subs[ns] = rospy.Subscriber(
                            topic,
                            FireEvent,
                            self._on_fire_event,
                            callback_args=ns,
                            queue_size=50,
                        )
                        rospy.loginfo("[referee] subscribed fire_event: %s", topic)

    def _on_robot_state(self, msg, ns):
        with self._lock:
            record = self._ensure_robot_record(ns)
            if record is None:
                return

            team_from_msg = self._decode_team_code(msg.team)
            team_from_ns = self._detect_team(ns)
            if team_from_msg in ("red", "blue"):
                prev_team = record.get("team", "unknown")
                if team_from_ns in ("red", "blue") and team_from_ns != team_from_msg:
                    rospy.logwarn_throttle(
                        2.0,
                        "[referee] team mismatch: ns=%s ns_team=%s msg_team=%s",
                        ns,
                        team_from_ns,
                        team_from_msg,
                    )
                if prev_team != team_from_msg:
                    rospy.loginfo(
                        "[referee] team updated by RobotState: ns=%s %s->%s",
                        ns,
                        prev_team,
                        team_from_msg,
                    )
                record["team"] = team_from_msg
            elif record.get("team", "unknown") not in ("red", "blue"):
                # msg.team 无法解析时，才回退到命名空间推断。
                record["team"] = team_from_ns

            record["x"] = float(msg.pose.position.x)
            record["y"] = float(msg.pose.position.y)
            record["yaw"] = float(self._quaternion_to_yaw(msg.pose.orientation))

    def _ray_hit(self, shooter_x, shooter_y, shooter_yaw, target_x, target_y):
        dx = float(target_x) - float(shooter_x)
        dy = float(target_y) - float(shooter_y)
        dist = math.hypot(dx, dy)
        if dist <= 1e-6 or dist >= self.fire_range:
            return False

        dir_x = math.cos(shooter_yaw)
        dir_y = math.sin(shooter_yaw)

        forward = dx * dir_x + dy * dir_y
        if forward <= 0.0:
            return False

        # 2D 叉积模长=到射线垂距（方向向量已单位化）
        perp = abs(dx * dir_y - dy * dir_x)
        return perp < self.hit_width

    def _segment_point_distance(self, x0, y0, x1, y1, px, py):
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 1e-9:
            return math.hypot(float(px) - float(x0), float(py) - float(y0)), 0.0

        t = ((float(px) - float(x0)) * dx + (float(py) - float(y0)) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        cx = float(x0) + t * dx
        cy = float(y0) + t * dy
        return math.hypot(float(px) - cx, float(py) - cy), t

    def _blocked_by_robot(self, shooter_ns, shooter_x, shooter_y, target_ns, target_x, target_y):
        width = float(self.occlusion_width)
        if width <= 0.0:
            return False

        for other_ns, state in self.global_states.items():
            if other_ns in (shooter_ns, target_ns):
                continue
            if not state.get("alive", True):
                continue

            ox = float(state.get("x", 0.0))
            oy = float(state.get("y", 0.0))
            dist, t = self._segment_point_distance(shooter_x, shooter_y, target_x, target_y, ox, oy)
            if dist <= width and t > 1e-3 and t < (1.0 - 1e-3):
                return True

        return False

    def _has_clear_shot(self, shooter_ns, shooter_x, shooter_y, target_ns, target_x, target_y):
        if not self._has_line_of_sight(shooter_x, shooter_y, target_x, target_y):
            return False
        if self._blocked_by_robot(shooter_ns, shooter_x, shooter_y, target_ns, target_x, target_y):
            return False
        return True
    def _world_to_map(self, x, y):
        """世界坐标 -> 栅格坐标 (mx,my)，失败返回 None"""
        if self._map_info is None:
            return None
        origin = self._map_info.origin.position
        res = float(self._map_info.resolution)
        mx = int((x - origin.x) / res)
        my = int((y - origin.y) / res)
        if mx < 0 or my < 0 or mx >= self._map_info.width or my >= self._map_info.height:
            return None
        return mx, my

    def _grid_index(self, mx, my):
        return my * self._map_info.width + mx

    def _cell_blocked(self, mx, my):
        """该栅格是否视为障碍"""
        idx = self._grid_index(mx, my)
        val = int(self._map_data[idx])
        if val < 0:
            return bool(self.block_unknown)
        return val >= self.occ_threshold

    def _bresenham(self, x0, y0, x1, y1):
        """Bresenham 栅格线算法，yield (x,y)"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            yield x, y
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _has_line_of_sight(self, x0, y0, x1, y1):
        """用 /map 判断两点之间是否无遮挡。没有地图时默认 True。"""
        with self._lock:
            if self._map_info is None or self._map_data is None:
                return True
            p0 = self._world_to_map(x0, y0)
            p1 = self._world_to_map(x1, y1)
            if p0 is None or p1 is None:
                # 在地图外：保守做法可以返回 False；想放宽可返回 True
                return False
            x0m, y0m = p0
            x1m, y1m = p1

            first = True
            for mx, my in self._bresenham(x0m, y0m, x1m, y1m):
                if first:
                    first = False
                    continue  # 跳过起点格（避免自己所在格被膨胀层/噪声误判）
                if self._cell_blocked(mx, my):
                    return False
            return True

    # ======================== 比赛生命周期管理 ========================

    def _init_match_stats(self):
        """为所有已发现的小车初始化对局统计。"""
        for ns in self.global_states:
            if ns not in self._match_stats:
                self._match_stats[ns] = {
                    "kills": 0,
                    "deaths": 0,
                    "shots_fired": 0,
                    "hits_landed": 0,
                    "hits_taken": 0,
                    "damage_dealt": 0.0,
                    "damage_taken": 0.0,
                    "death_time": None,
                }

    def _reset_match_state(self):
        """重置比赛状态，准备下一局。"""
        self._match_status = "IDLE"
        self._match_start_wall = 0.0
        self._match_winner = ""
        self._match_reason = ""
        self._match_ended = False
        self._match_stats = {}
        self._narrative = []
        self._match_start_str = ""
        self._match_end_str = ""

    def _try_auto_start_match(self):
        """检测双方是否都有机器人存活，自动开始比赛。"""
        if self._match_status != "IDLE":
            return

        red_alive = 0
        blue_alive = 0
        for ns, state in self.global_states.items():
            alive = bool(state.get("alive", True) and int(state.get("hp", 0)) > 0)
            if not alive:
                continue
            team = state.get("team", "")
            if team == "red":
                red_alive += 1
            elif team == "blue":
                blue_alive += 1

        if red_alive > 0 and blue_alive > 0:
            self._match_id += 1
            self._match_status = "PLAYING"
            self._match_start_wall = time.time()
            self._match_start_str = time.strftime("%Y-%m-%d %H:%M:%S")
            self._match_winner = ""
            self._match_reason = ""
            self._match_ended = False
            self._init_match_stats()
            self._narrative = []
            self._init_narrative_path()
            self._clear_log_files()

            self._append_narrative(
                "referee",
                "MATCH %d STARTED — red=%d alive, blue=%d alive, time_limit=%.1fs"
                % (self._match_id, red_alive, blue_alive, self.time_limit),
            )
            rospy.loginfo(
                "[referee] ===== MATCH %d STARTED ===== red=%d alive, blue=%d alive, time_limit=%.1fs",
                self._match_id,
                red_alive,
                blue_alive,
                self.time_limit,
            )

    def _check_match_end(self):
        """检查比赛是否应结束。"""
        if self._match_status != "PLAYING":
            return False
        if self._match_ended:
            return True

        red_alive = 0
        blue_alive = 0
        red_total = 0
        blue_total = 0
        for ns, state in self.global_states.items():
            team = state.get("team", "")
            alive = bool(state.get("alive", True) and int(state.get("hp", 0)) > 0)
            if team == "red":
                red_total += 1
                if alive:
                    red_alive += 1
            elif team == "blue":
                blue_total += 1
                if alive:
                    blue_alive += 1

        elapsed = time.time() - self._match_start_wall

        # 诊断：每 5 秒打印一次存活状态
        rospy.loginfo_throttle(
            5.0,
            "[referee] match alive check: red=%d/%d alive, blue=%d/%d alive, elapsed=%.1fs",
            red_alive, red_total, blue_alive, blue_total, elapsed,
        )

        # 诊断：每 10 秒打印一次各车 HP/弹药
        if red_total + blue_total > 0:
            ammo_lines = []
            for ns, state in sorted(self.global_states.items()):
                team = state.get("team", "?")
                hp = int(state.get("hp", 0))
                ammo = float(state.get("ammo", 0))
                alive = "A" if (state.get("alive", True) and hp > 0) else "D"
                ammo_lines.append("%s[%s] HP=%d AM=%.0f" % (ns, alive, hp, ammo))
            rospy.loginfo_throttle(
                10.0,
                "[referee] status: %s",
                " | ".join(ammo_lines),
            )

        # 条件1：一方全灭
        if red_alive == 0 and blue_alive > 0:
            self._match_winner = "blue"
            self._match_reason = "all_enemy_dead"
            self._end_match()
            return True
        if blue_alive == 0 and red_alive > 0:
            self._match_winner = "red"
            self._match_reason = "all_enemy_dead"
            self._end_match()
            return True
        if red_alive == 0 and blue_alive == 0 and red_total > 0 and blue_total > 0:
            self._match_winner = "draw"
            self._match_reason = "mutual_destruction"
            self._end_match()
            return True

        # 条件2：超时
        if self.time_limit > 0 and elapsed >= self.time_limit:
            if red_alive > blue_alive:
                self._match_winner = "red"
            elif blue_alive > red_alive:
                self._match_winner = "blue"
            else:
                self._match_winner = "draw"
            self._match_reason = "timeout"
            self._end_match()
            return True

        # 条件3：弹药耗尽僵局 — 所有存活小车弹药都为 0，继续打也没意义
        if red_alive > 0 and blue_alive > 0:
            all_out_of_ammo = True
            for ns, state in self.global_states.items():
                if not state.get("alive", True):
                    continue
                if int(state.get("hp", 0)) <= 0:
                    continue
                if float(state.get("ammo", 0)) > 0:
                    all_out_of_ammo = False
                    break
            if all_out_of_ammo:
                rospy.logwarn(
                    "[referee] all alive robots out of ammo — ending match as draw"
                )
                self._match_winner = "draw"
                self._match_reason = "ammo_starvation"
                self._end_match()
                return True

        return False

    def _end_match(self):
        """结束比赛，生成记录并发布。"""
        self._match_ended = True
        self._match_status = "FINISHED"
        self._match_end_str = time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self._match_start_wall

        # 补充存活时间统计
        for ns in self.global_states:
            stat = self._match_stats.get(ns)
            if stat is None:
                continue
            if stat["death_time"] is None:
                stat["survival_time"] = elapsed
            else:
                stat["survival_time"] = stat["death_time"] - self._match_start_wall

        rospy.loginfo(
            "[referee] ===== MATCH %d ENDED ===== winner=%s reason=%s duration=%.1fs",
            self._match_id,
            self._match_winner,
            self._match_reason,
            elapsed,
        )

        # 记录比赛结束叙事
        self._append_narrative(
            "referee",
            "MATCH %d ENDED — winner=%s reason=%s duration=%.1fs"
            % (self._match_id, self._match_winner, self._match_reason, elapsed),
        )

        # 立即发布 FINISHED 状态，让 Manager 尽快收到
        self._publish_game_state()

        # 让所有小车原地停止
        self._stop_all_robots()

        # 构建并发布 MatchRecord
        record = self._build_match_record(elapsed)
        self.game_result_pub.publish(record)

        # 保存日志
        self._save_match_log(elapsed)

        # 10 秒后关掉整个仿真（让操作者看到最终状态）
        rospy.loginfo("[referee] match done — shutting down in 10s (Ctrl+C to skip)")
        rospy.Timer(rospy.Duration(10.0), lambda event: rospy.signal_shutdown("match_ended"), oneshot=True)

    def _stop_all_robots(self):
        """向所有小车发布 STOP 命令，覆盖 Manager 后续任务。"""
        for ns in self.global_states:
            if ns not in self._stop_pubs:
                topic = "/%s/task_cmd" % ns
                self._stop_pubs[ns] = rospy.Publisher(topic, TaskCommand, queue_size=10)

            cmd = TaskCommand()
            cmd.task_id = 0
            cmd.action_type = "STOP"
            cmd.timeout = 0.0
            cmd.reason = "match_ended"
            self._stop_pubs[ns].publish(cmd)

        rospy.loginfo(
            "[referee] STOP sent to %d robots", len(self.global_states)
        )

    def _build_match_record(self, elapsed):
        """构建 MatchRecord 消息。"""
        record = MatchRecord()
        record.match_id = self._match_id
        record.winner = str(self._match_winner)
        record.reason = str(self._match_reason)
        record.duration = float(elapsed)
        record.time_limit = float(self.time_limit)
        record.red_config = ""
        record.blue_config = ""

        record.red_stats = self._build_team_match_stat("red", elapsed)
        record.blue_stats = self._build_team_match_stat("blue", elapsed)

        return record

    def _build_team_match_stat(self, team, elapsed):
        """构建单方阵营的对局统计。"""
        msg = TeamMatchStat()
        msg.team = str(team)

        total_kills = 0
        total_deaths = 0
        total_shots = 0
        total_hits = 0
        total_dmg_dealt = 0.0
        total_dmg_taken = 0.0

        for ns in sorted(self.global_states.keys()):
            state = self.global_states.get(ns, {})
            if state.get("team") != team:
                continue

            stat = self._match_stats.get(ns, {})
            robot = RobotMatchStat()
            robot.robot_ns = str(ns)
            robot.final_hp = float(state.get("hp", 0))
            robot.final_ammo = float(state.get("ammo", 0))
            robot.was_alive = bool(state.get("alive", False) and int(state.get("hp", 0)) > 0)
            robot.kills = int(stat.get("kills", 0))
            robot.deaths = int(stat.get("deaths", 0))
            robot.shots_fired = int(stat.get("shots_fired", 0))
            robot.hits_landed = int(stat.get("hits_landed", 0))
            robot.hits_taken = int(stat.get("hits_taken", 0))
            robot.damage_dealt = float(stat.get("damage_dealt", 0.0))
            robot.damage_taken = float(stat.get("damage_taken", 0.0))
            robot.survival_time = float(stat.get("survival_time", elapsed))

            msg.robots.append(robot)

            total_kills += robot.kills
            total_deaths += robot.deaths
            total_shots += robot.shots_fired
            total_hits += robot.hits_landed
            total_dmg_dealt += robot.damage_dealt
            total_dmg_taken += robot.damage_taken

        msg.total_kills = total_kills
        msg.total_deaths = total_deaths
        msg.total_shots_fired = total_shots
        msg.total_hits_landed = total_hits
        msg.total_damage_dealt = total_dmg_dealt
        msg.total_damage_taken = total_dmg_taken

        return msg

    def _save_match_log(self, elapsed):
        """比赛结束日志（已由 narrative.jsonl 实时写入）。"""
        if self._narrative_path:
            rospy.loginfo(
                "[referee] narrative file: %s (%d lines)",
                self._narrative_path, len(self._narrative),
            )

    @staticmethod
    def _team_stat_to_dict(team_msg):
        """将 TeamMatchStat 消息转为 dict。"""
        robots = []
        for r in team_msg.robots:
            robots.append({
                "robot_ns": r.robot_ns,
                "final_hp": r.final_hp,
                "final_ammo": r.final_ammo,
                "was_alive": r.was_alive,
                "kills": r.kills,
                "deaths": r.deaths,
                "shots_fired": r.shots_fired,
                "hits_landed": r.hits_landed,
                "hits_taken": r.hits_taken,
                "damage_dealt": r.damage_dealt,
                "damage_taken": r.damage_taken,
                "survival_time": r.survival_time,
            })
        return {
            "team": team_msg.team,
            "total_kills": team_msg.total_kills,
            "total_deaths": team_msg.total_deaths,
            "total_shots_fired": team_msg.total_shots_fired,
            "total_hits_landed": team_msg.total_hits_landed,
            "total_damage_dealt": team_msg.total_damage_dealt,
            "total_damage_taken": team_msg.total_damage_taken,
            "robots": robots,
        }

    def _on_game_command(self, msg):
        """处理 /game/command 的外部控制。"""
        cmd = str(msg.data).strip().lower()
        with self._lock:
            if cmd == "start":
                if self._match_status == "IDLE":
                    # 强制开始
                    self._match_id += 1
                    self._match_status = "PLAYING"
                    self._match_start_wall = time.time()
                    self._match_start_str = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._match_winner = ""
                    self._match_reason = ""
                    self._match_ended = False
                    self._init_match_stats()
                    self._narrative = []
                    self._init_narrative_path()
                    self._clear_log_files()
                    self._append_narrative(
                        "referee",
                        "MATCH %d FORCE STARTED via /game/command" % self._match_id,
                    )
                    rospy.loginfo("[referee] ===== MATCH %d FORCE STARTED via /game/command =====", self._match_id)
                else:
                    rospy.logwarn("[referee] cannot start: status=%s", self._match_status)

            elif cmd == "stop":
                if self._match_status == "PLAYING" and not self._match_ended:
                    self._match_winner = ""
                    self._match_reason = "manual_stop"
                    self._end_match()
                else:
                    rospy.logwarn("[referee] cannot stop: status=%s", self._match_status)

            elif cmd == "reset":
                self._reset_match_state()
                rospy.loginfo("[referee] match state reset via /game/command")

            else:
                rospy.logwarn("[referee] unknown game command: %s", cmd)

    def _shorten_ns(self, ns):
        """简化命名空间：robot_red -> red, robot_blue2 -> blue2"""
        s = str(ns).strip()
        for prefix in ("robot_", "robot"):
            if s.startswith(prefix):
                return s[len(prefix):]
        return s

    def _append_narrative(self, source, message):
        """追加一条叙事到文件，格式：{"t": 秒, "m": "消息"}
        仅记录 combat 事件（START/END/hit/kill/miss），不含命令。
        """
        elapsed = time.time() - self._match_start_wall if self._match_start_wall > 0 else 0.0
        # 统一转 unicode
        try:
            if isinstance(message, str):
                message = message.decode("utf-8")
            if isinstance(source, str):
                source = source.decode("utf-8")
        except NameError:
            pass
        # OrderedDict 确保 json 输出时 t 在前
        obj = OrderedDict()
        obj["t"] = round(elapsed, 2)
        obj["m"] = "[%s] %s" % (source, message)
        line = json.dumps(obj)
        self._narrative.append(line)
        if self._narrative_path:
            try:
                with open(self._narrative_path, "a") as f:
                    f.write(line + "\n")
            except Exception as exc:
                rospy.logwarn("[referee] write narrative failed: %s", exc)

    def _append_team_log(self, team, record):
        """追加一条队伍日志到对应文件。
        :param team: "red" 或 "blue"
        :param record: dict，含 t 等字段
        """
        path = self._team_log_paths.get(team)
        if not path:
            return
        elapsed = time.time() - self._match_start_wall if self._match_start_wall > 0 else 0.0
        obj = OrderedDict()
        obj["t"] = round(elapsed, 2)
        obj.update(record)
        line = json.dumps(obj)
        try:
            with open(path, "a") as f:
                f.write(line + "\n")
        except Exception as exc:
            rospy.logwarn("[referee] write team log (%s) failed: %s", team, exc)

    def _on_narrative_event(self, msg):
        """接收 Manager 等节点发来的叙事（纯文本）。
        命令写入对应队伍日志，不写入 narrative。
        """
        raw = msg.data
        try:
            if isinstance(raw, str):
                raw = raw.decode("utf-8")
        except Exception:
            pass
        # 判断消息属于哪队：看 ns 中是否包含 red/blue
        raw_lower = raw.lower()
        team = None
        if "red" in raw_lower:
            team = "red"
        elif "blue" in raw_lower:
            team = "blue"
        if team:
            self._append_team_log(team, {"event": "command", "msg": raw})

    def _publish_game_state(self):
        """发布当前比赛状态。"""
        msg = GameState()
        msg.status = str(self._match_status)
        if self._match_start_wall > 0:
            msg.elapsed = float(time.time() - self._match_start_wall)
        else:
            msg.elapsed = 0.0
        msg.time_limit = float(self.time_limit)
        msg.winner = str(self._match_winner)
        msg.reason = str(self._match_reason)
        self.game_state_pub.publish(msg)

    # ======================== 开火事件处理（原有逻辑 + 统计）=======================

    def _on_fire_event(self, msg, topic_ns):
        shooter_ns = self._normalize_ns(msg.shooter_ns) or self._normalize_ns(topic_ns)
        if not shooter_ns:
            return

        with self._lock:
            shooter = self._ensure_robot_record(shooter_ns)
            if shooter is None:
                return

            shooter_team = shooter.get("team", "unknown")
            if shooter_team not in ("red", "blue"):
                rospy.logwarn("[referee] fire from unknown team: %s team=%s", shooter_ns, shooter_team)
                return

            # 打印每条 fire_event 的抵达记录（不节流）
            shooter_hp = int(shooter.get("hp", 0))
            shooter_ammo = float(shooter.get("ammo", 0))
            shooter_alive = bool(shooter.get("alive", True) and shooter_hp > 0)
            rospy.loginfo(
                "[referee] FIRE_EVENT from=%s team=%s alive=%s hp=%d ammo=%.0f pos=(%.2f,%.2f)",
                shooter_ns, shooter_team, shooter_alive,
                shooter_hp, shooter_ammo,
                float(msg.x), float(msg.y),
            )

            # 开火先进行弹药结算：无弹药则拦截，命中判定不再继续。
            if not shooter_alive:
                rospy.logwarn("[referee] FIRE_BLOCKED (dead): %s", shooter_ns)
                return

            old_ammo = float(shooter.get("ammo", self.default_ammo))
            if old_ammo <= 0.0:
                rospy.logwarn("[referee] FIRE_BLOCKED (no ammo): %s", shooter_ns)
                return
            shooter["ammo"] = max(0.0, old_ammo - 1.0)

            # 以 fire_event 的位姿作为射击真值。
            shooter["x"] = float(msg.x)
            shooter["y"] = float(msg.y)
            shooter["yaw"] = float(msg.yaw)

            # 统计：记录开枪次数
            shooter_stat = self._match_stats.get(shooter_ns)
            if shooter_stat is not None:
                shooter_stat["shots_fired"] = shooter_stat.get("shots_fired", 0) + 1

            enemy_team = "blue" if shooter_team == "red" else "red"
            hit_any = False
            for enemy_ns, enemy in self.global_states.items():
                if enemy_ns == shooter_ns:
                    continue
                if enemy.get("team") != enemy_team:
                    continue
                if not enemy.get("alive", True):
                    continue

                if self._ray_hit(
                        shooter["x"],
                        shooter["y"],
                        shooter["yaw"],
                        enemy.get("x", 0.0),
                        enemy.get("y", 0.0),
                    ):
                        if not self._has_clear_shot(
                            shooter_ns,
                            shooter["x"],
                            shooter["y"],
                            enemy_ns,
                            enemy.get("x", 0.0),
                            enemy.get("y", 0.0),
                        ):
                            rospy.loginfo(
                                "[referee] ray hit %s but blocked (occlusion/LOS)", enemy_ns
                            )
                            continue

                        hit_any = True
                        old_hp = int(enemy.get("hp", self.default_hp))
                        new_hp = max(0, old_hp - self.fire_damage)
                        enemy["hp"] = new_hp
                        enemy["alive"] = bool(new_hp > 0)

                        # 统计：命中 + 伤害
                        damage = old_hp - new_hp
                        if shooter_stat is not None:
                            shooter_stat["hits_landed"] = shooter_stat.get("hits_landed", 0) + 1
                            shooter_stat["damage_dealt"] = shooter_stat.get("damage_dealt", 0.0) + float(damage)
                        enemy_stat = self._match_stats.get(enemy_ns)
                        if enemy_stat is not None:
                            enemy_stat["hits_taken"] = enemy_stat.get("hits_taken", 0) + 1
                            enemy_stat["damage_taken"] = enemy_stat.get("damage_taken", 0.0) + float(damage)

                        rospy.loginfo(
                            "[referee] HIT: shooter=%s target=%s hp:%d->%d",
                            shooter_ns,
                            enemy_ns,
                            old_hp,
                            new_hp,
                        )

                        if old_hp > 0 and new_hp == 0:
                            # 统计：击杀 / 死亡
                            if shooter_stat is not None:
                                shooter_stat["kills"] = shooter_stat.get("kills", 0) + 1
                            if enemy_stat is not None:
                                enemy_stat["deaths"] = enemy_stat.get("deaths", 0) + 1
                                enemy_stat["death_time"] = time.time()

                            # 记录击杀叙事（简化名+坐标+弹药量）
                            s_name = self._shorten_ns(shooter_ns)
                            e_name = self._shorten_ns(enemy_ns)
                            sx = float(shooter.get("x", 0))
                            sy = float(shooter.get("y", 0))
                            ex = float(enemy.get("x", 0))
                            ey = float(enemy.get("y", 0))
                            self._append_narrative(
                                "referee",
                                "%s(%.1f,%.1f) KILLED %s(%.1f,%.1f) (hp:%d->0 ammo=%d)"
                                % (s_name, sx, sy, e_name, ex, ey, old_hp, int(shooter["ammo"])),
                            )
                            # 队伍日志：攻击方击杀，受击方阵亡
                            self._append_team_log(shooter_team, {
                                "event": "attack", "shooter": s_name, "pos": [round(sx,1), round(sy,1)],
                                "target": e_name, "hit": True, "killed": True,
                            })
                            self._append_team_log(enemy_team, {
                                "event": "hit", "target": e_name, "pos": [round(ex,1), round(ey,1)],
                                "attacker": s_name, "killed": True,
                            })
                            rospy.loginfo("[referee] KILL: shooter=%s target=%s", shooter_ns, enemy_ns)

                        else:
                            # 命中但不致死（简化名+坐标+弹药量）
                            s_name = self._shorten_ns(shooter_ns)
                            e_name = self._shorten_ns(enemy_ns)
                            sx = float(shooter.get("x", 0))
                            sy = float(shooter.get("y", 0))
                            ex = float(enemy.get("x", 0))
                            ey = float(enemy.get("y", 0))
                            self._append_narrative(
                                "referee",
                                "%s(%.1f,%.1f) hits %s(%.1f,%.1f) (hp:%d->%d ammo=%d)"
                                % (s_name, sx, sy, e_name, ex, ey, old_hp, new_hp, int(shooter["ammo"])),
                            )
                            # 队伍日志：攻击命中，受击
                            self._append_team_log(shooter_team, {
                                "event": "attack", "shooter": s_name, "pos": [round(sx,1), round(sy,1)],
                                "target": e_name, "hit": True, "killed": False,
                            })
                            self._append_team_log(enemy_team, {
                                "event": "hit", "target": e_name, "pos": [round(ex,1), round(ey,1)],
                                "attacker": s_name, "killed": False,
                            })

            if not hit_any:
                # 开枪但全部 miss（简化名+坐标+弹药量）
                s_name = self._shorten_ns(shooter_ns)
                sx = float(shooter.get("x", 0))
                sy = float(shooter.get("y", 0))
                self._append_narrative(
                    "referee",
                    "%s(%.1f,%.1f) fires — missed all enemies (ammo=%d)"
                    % (s_name, sx, sy, int(shooter["ammo"])),
                )
                # 队伍日志：攻击未命中
                self._append_team_log(shooter_team, {
                    "event": "attack", "shooter": s_name, "pos": [round(sx,1), round(sy,1)],
                    "hit": False,
                })
                rospy.loginfo(
                    "[referee] fire from %s missed all enemies (ray_hit or LOS blocked)",
                    shooter_ns,
                )

    def _angle_diff(self, a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))
    
    def _build_visible_enemies(self, observer_team):
        enemy_team = "blue" if observer_team == "red" else "red"

        friendlies = []
        enemies = []
        for ns, state in self.global_states.items():
            if not state.get("alive", True):
                continue
            if state.get("team") == observer_team:
                friendlies.append((ns, state))
            elif state.get("team") == enemy_team:
                enemies.append((ns, state))

        visible = []
        half_fov = 0.5 * self.fov_rad
        for enemy_ns, enemy_state in enemies:
            ex = float(enemy_state.get("x", 0.0))
            ey = float(enemy_state.get("y", 0.0))

            seen = False
            for _, friendly_state in friendlies:
                fx = float(friendly_state.get("x", 0.0))
                fy = float(friendly_state.get("y", 0.0))
                fyaw = float(friendly_state.get("yaw", 0.0))

                dist = math.hypot(ex - fx, ey - fy)
                if dist > self.vision_range:
                    continue

                bearing = math.atan2(ey - fy, ex - fx)
                if abs(self._angle_diff(bearing, fyaw)) > half_fov:
                    continue

                if not self._has_line_of_sight(fx, fy, ex, ey):
                    continue
                seen = True
                break

            if seen:
                info = EnemyInfo()
                info.robot_ns = enemy_ns
                info.x = ex
                info.y = ey
                info.hp = int(enemy_state.get("hp", self.default_hp))
                visible.append(info)

        msg = VisibleEnemies()
        msg.enemies = visible
        return msg

    def _publish_visible_enemies(self):
        with self._lock:
            red_msg = self._build_visible_enemies("red")
            blue_msg = self._build_visible_enemies("blue")

        self.red_enemy_pub.publish(red_msg)
        self.blue_enemy_pub.publish(blue_msg)

    def _build_team_macro_state(self, team):
        msg = TeamMacroState()
        msg.team = str(team)

        total_hp = 0
        total_ammo = 0.0
        alive_count = 0
        dead_count = 0

        for ns in sorted(self.global_states.keys()):
            state = self.global_states.get(ns, {})
            if state.get("team") != team:
                continue

            hp = int(state.get("hp", self.default_hp))
            ammo = float(state.get("ammo", self.default_ammo))
            alive = bool(state.get("alive", True) and hp > 0)

            msg.robot_ns.append(ns)
            msg.hp.append(hp)
            msg.ammo.append(ammo)
            msg.alive.append(alive)

            total_hp += hp
            total_ammo += ammo
            if alive:
                alive_count += 1
            else:
                dead_count += 1

        msg.total_hp = int(total_hp)
        msg.total_ammo = float(total_ammo)
        msg.alive_count = int(alive_count)
        msg.dead_count = int(dead_count)
        return msg

    def _publish_macro_state(self):
        with self._lock:
            red = self._build_team_macro_state("red")
            blue = self._build_team_macro_state("blue")

        msg = BattleMacroState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.red = red
        msg.blue = blue
        self.macro_state_pub.publish(msg)

    def run(self):
        main_rate = rospy.Rate(self.loop_hz)
        discover_interval = 1.0 / self.discover_hz if self.discover_hz > 0.0 else 1.0
        last_discover = 0.0
        game_state_interval = 1.0  # 1 Hz
        last_game_state = 0.0

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            if now - last_discover >= discover_interval:
                self._discover_and_subscribe()
                last_discover = now

            with self._lock:
                # 比赛生命周期管理
                self._try_auto_start_match()
                self._check_match_end()

                # 比赛结束后持续发 STOP，压制 Manager 后续发来的任务
                if self._match_status == "FINISHED":
                    self._stop_all_robots()

            self._publish_visible_enemies()
            self._publish_macro_state()

            if now - last_game_state >= game_state_interval:
                with self._lock:
                    self._publish_game_state()
                last_game_state = now

            main_rate.sleep()

    def _on_map(self, msg):
        with self._lock:
            self._map_info = msg.info
            self._map_data = msg.data  # tuple/list of int8

def main():
    rospy.init_node("referee_node", anonymous=False)

    node = RefereeNode()
    node.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
