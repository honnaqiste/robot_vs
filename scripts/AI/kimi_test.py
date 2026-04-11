#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 旧版本写法 (openai < 1.0)
import openai

# 配置（不需要初始化 OpenAI 类）
openai.api_key = "你的Kimi API Key"
openai.base_url = "https://api.moonshot.cn/v1"

# 调用方式
response = openai.chat.completions.create(
    model="kimi-k2.5",
    messages=[
        {"role": "system", "content": "你是科研专家"},
        {"role": "user", "content": "你好"}
    ],
    temperature=0.3
)

# 输出结果
print(response.choices[0].message.content)