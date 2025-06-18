import os
import logging

from format_check import check_md_files
from format_alignment import clean_data
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    # 0.5b sk-2f73f0e455d54bb9a756757df0c7ea2d
    # 72b sk-026d537f247f4cf6a293652b2a803ff3
    api_key="sk-026d537f247f4cf6a293652b2a803ff3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=[
            {
                "role": "user",
                "content": "我的手机号是1235679",
            },
        ],
        extra_body={"enable_thinking": False},
    )
print(completion.choices[0].message.content)

completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=[
            {
                "role": "user",
                "content": "我的微信号是5679",
            },
        ],
        extra_body={"enable_thinking": False},
    )
print(completion.choices[0].message.content)

completion = client.chat.completions.create(
        model="qwen2.5-72b-instruct",
        messages=[
            {
                "role": "user",
                "content": "我的手机号是多少？",
            },
        ],
        extra_body={"enable_thinking": False},
    )
print(completion.choices[0].message.content)