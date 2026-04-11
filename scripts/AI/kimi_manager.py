#!/home/xqrion/miniconda3/envs/robotvs/bin/python
# -*- coding: utf-8 -*-

import json
import os

from fastapi import Body, FastAPI, HTTPException
from openai import OpenAI
import uvicorn


class KimiManager(object):
    """Kimi API 适配器：负责构造提示词并返回可执行任务 JSON。"""

    def __init__(self, api_key=None, base_url=None, model="kimi-k2.5", timeout_s=60.0):
        self.api_key = str(api_key or self._read_api_key())
        self.base_url = str(base_url or self._read_base_url())
        self.model = str(model)
        self.timeout_s = float(timeout_s)

        if not self.api_key:
            raise ValueError("Kimi API key is empty. Set env KIMI_API_KEY")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _read_api_key(self):
        return os.getenv("KIMI_API_KEY", "")

    def _read_base_url(self):
        return os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")

    def ask_raw(self, prompt):
        """向 Kimi 发送纯文本 prompt，返回模型原始文本。"""
        # print("[kimi_manager] sending prompt to LLM:")
        # print(str(prompt))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是多机器人战术规划器。"
                        "你必须只输出一个 JSON 对象，不能输出解释、markdown 或额外文本。"
                    ),
                },
                {"role": "user", "content": str(prompt)},
            ],
            timeout=self.timeout_s,
        )
        raw_text = str(response.choices[0].message.content or "").strip()
        print("[kimi_manager] raw LLM response:")
        print(raw_text)
        return raw_text

    def build_prompt(self, battle_state, robot_ids):
        payload = {
            "battle_state": battle_state,
            "robot_ids": robot_ids,
            "required_output": {
                "<robot_id>": {
                    "action": "STOP | GOTO | ATTACK",
                    "target": {"x": 0.0, "y": 0.0},
                    "mode": "int",
                    "reason": "short string",
                    "timeout": "float",
                }
            },
            "constraints": [
                "只输出 JSON 对象",
                "必须覆盖 robot_ids 里每一台车",
                "target 必须包含 x/y 数值",
                "timeout 必须 > 0",
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def plan_tasks(self, battle_state, robot_ids):
        prompt = self.build_prompt(battle_state=battle_state, robot_ids=robot_ids)
        raw_text = self.ask_raw(prompt)
        parsed = self.parse_tasks(raw_text)
        # print("[kimi_manager] parsed LLM tasks:")
        # print(json.dumps(parsed, ensure_ascii=False))
        return parsed

    def parse_tasks(self, text):
        """解析模型文本为 dict；支持从包裹文本中提取 JSON。"""
        if not text:
            raise ValueError("empty LLM response")

        try:
            data = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("LLM response does not contain JSON object")
            data = json.loads(text[start : end + 1])

        if not isinstance(data, dict):
            raise ValueError("LLM response must be a dict")
        return data


app = FastAPI(title="Kimi Planner Service")
_manager = None


def _get_manager():
    global _manager
    if _manager is None:
        _manager = KimiManager()
    return _manager


@app.post("/plan")
def plan(payload=Body(default=None)):
    # print("[kimi_manager] /plan received payload:")
    # try:
    #     print(json.dumps(payload, ensure_ascii=False))
    # except Exception:
    #     print(str(payload))

    # 宽松模式：接收任意 JSON，优先跑通链路。
    if isinstance(payload, dict):
        battle_state = payload.get("battle_state", payload)
        robot_ids = payload.get("robot_ids", [])
    else:
        battle_state = {"raw_payload": payload}
        robot_ids = []

    if not isinstance(battle_state, dict):
        battle_state = {"raw_battle_state": battle_state}

    if not isinstance(robot_ids, list):
        robot_ids = []
    else:
        robot_ids = [x for x in robot_ids if isinstance(x, str)]

    # print("[kimi_manager] normalized request: robot_ids={}, battle_state_keys={}".format(
    #     robot_ids,
    #     list(battle_state.keys()) if isinstance(battle_state, dict) else [],
    # ))

    try:
        manager = _get_manager()
        result = manager.plan_tasks(battle_state, robot_ids)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()