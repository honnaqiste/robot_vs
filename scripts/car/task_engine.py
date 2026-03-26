#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading

import rospy
from robot_vs.msg import TaskCommand

from skills.base_skill import RUNNING, SUCCESS, FAILED


class TaskEngine(object):
    """Maintain current task and drive skill execution.

    Responsibilities:
    1) Keep current task snapshot received from car_node.
    2) Instantiate the appropriate skill via SkillManager on task change.
    3) On every tick(), call the current skill's update() and handle
       transitions (RUNNING → SUCCESS / FAILED).
    4) Monitor task timeout and fail-safe to STOP.
    """

    def __init__(self, ns, skill_manager):
        self.ns = str(ns)
        self.skill_manager = skill_manager

        self._lock = threading.RLock()
        self._current_task = None   # latest task dict
        self._task_status = "IDLE"  # IDLE / RUNNING / SUCCESS / FAILED
        self._current_action = "NONE"
        self._task_start_t = None

        rospy.loginfo("[%s] TaskEngine initialised", self.ns)

    # ------------------------------------------------------------------
    # Task receiver
    # ------------------------------------------------------------------

    def accept_task(self, msg):
        if not isinstance(msg, TaskCommand):
            raise ValueError("accept_task expects TaskCommand")

        with self._lock:
            current = self._current_task
            if current is not None and int(current.get("task_id", 0)) == int(msg.task_id):
                return  # same task, ignore duplicate

            rospy.loginfo(
                "[%s] TaskEngine: new task task_id=%d action=%s target=(%.2f, %.2f)",
                self.ns, msg.task_id, msg.action_type, msg.target_x, msg.target_y,
            )

            task_dict = {
                "task_id": msg.task_id,
                "action_type": msg.action_type,
                "target_x": msg.target_x,
                "target_y": msg.target_y,
                "mode": msg.mode,
                "reason": msg.reason,
                "timeout": msg.timeout,
            }
            try:
                self.skill_manager.switch_skill(msg.action_type, task_dict)
            except Exception as exc:
                rospy.logwarn("[%s] switch_skill failed: %s", self.ns, exc)

            self._current_task = task_dict
            self._task_start_t = rospy.Time.now().to_sec()
            self._task_status = RUNNING
            self._current_action = str(msg.action_type).upper()
            self.skill_manager.set_task_feedback(
                task_id=int(msg.task_id),
                current_action=self._current_action,
                task_status=self._task_status,
                mode=int(msg.mode),
            )

    # ------------------------------------------------------------------
    # Main loop step
    # ------------------------------------------------------------------

    def tick(self):
        """Called every loop iteration from car_node."""
        with self._lock:
            task = self._current_task
            task_status = self._task_status
            task_start_t = self._task_start_t

        if task is None:
            self.skill_manager.set_task_feedback(
                task_id=0,
                current_action="NONE",
                task_status="IDLE",
                mode=0,
            )
            return

        if task_status == RUNNING and self._is_task_timeout(task, task_start_t):
            rospy.logwarn("[%s] task timeout: task_id=%s", self.ns, task.get("task_id"))
            with self._lock:
                self._task_status = FAILED
            self.skill_manager.stop_active_skill()
            try:
                self.skill_manager.switch_skill("STOP", self._build_timeout_stop_task(task))
            except Exception as exc:
                rospy.logwarn("[%s] timeout stop switch failed: %s", self.ns, exc)
            self.skill_manager.set_task_feedback(
                task_id=int(task.get("task_id", 0)),
                current_action=str(task.get("action_type", "NONE")).upper(),
                task_status=FAILED,
                mode=int(task.get("mode", 0)),
            )
            return

        try:
            result = self.skill_manager.update_active_skill()
        except Exception as exc:
            rospy.logwarn("[%s] skill.update() raised: %s", self.ns, exc)
            result = FAILED

        with self._lock:
            self._task_status = result

        self.skill_manager.set_task_feedback(
            task_id=int(task.get("task_id", 0)),
            current_action=self._current_action,
            task_status=self._task_status,
            mode=int(task.get("mode", 0)),
        )

    def _is_task_timeout(self, task, task_start_t):
        if task_start_t is None:
            return False

        timeout = float(task.get("timeout", 0.0) or 0.0)
        if timeout <= 0.0:
            return False

        elapsed = rospy.Time.now().to_sec() - float(task_start_t)
        return elapsed > timeout

    def _build_timeout_stop_task(self, task):
        return {
            "task_id": int(task.get("task_id", 0)),
            "action_type": "STOP",
            "target_x": 0.0,
            "target_y": 0.0,
            "mode": 0,
            "reason": "timeout",
            "timeout": 0.0,
        }
