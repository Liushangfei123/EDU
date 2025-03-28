import os
import yaml
import sys
import json
from typing import Annotated, Literal

from autogen import ConversableAgent
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, register_function
from logger import Logger
from utils import load_yaml, load_ZhiPu
from functions import function_map
from tqdm import tqdm
from RAG.rag_utils import create_collection, upload_txt_file, rag_search

# 定义操作符类型
Operator = Literal["+", "-", "*", "/"]

config = load_yaml("config.yaml")
config_list_zhipu = load_ZhiPu(config)

def calculator(a: int, b: int, operator: Annotated[Operator, "operator"]) -> int:
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")

# 自定义代理类，重写 send 方法确保每个消息中包含 prompt 字段
class MyConversableAgent(ConversableAgent):
    def send(self, message, recipient, silent=False):
        # 如果消息是字符串，将其包装为字典
        if isinstance(message, str):
            message = {"content": message, "prompt": message}
        # 如果消息是字典但缺少 prompt，则补充
        elif isinstance(message, dict) and "prompt" not in message:
            message["prompt"] = message.get("content", "")
        super().send(message, recipient, silent)

# 定义助手代理，增加终止提示以防止无限递归
assistant = MyConversableAgent(
    name="Assistant",
    system_message=(
        "You are a helpful AI assistant. You can help with simple calculations. "
        "When your task is complete, please provide the final answer and include the word 'TERMINATE' to end the conversation."
    ),
    llm_config={
        "config_list": [{
            "model": "GLM-4-Flash",
            "api_key": "398f5e33f62c6a9dc7cd95987b14aae6.rKQve5A1VUNNHW4Z",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "price": [0.0, 0.0]  # 添加 price 字段以满足配置要求
        }]
    },
)

# 定义用户代理，同样使用自定义代理类
user_proxy = MyConversableAgent(
    name="User",
    llm_config=False,
    # 增加对“TERMINATE”的检测，保证一旦完成任务对话能结束
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

# 注册计算器工具
assistant.register_for_llm(name="calculator", description="A simple calculator")(calculator)
user_proxy.register_for_execution(name="calculator")(calculator)

# 发起对话
chat_result = user_proxy.initiate_chat(
    assistant,
    message="What is (44232 + 13312 / (232 - 32)) * 5?"
)

print("最终对话结果：", chat_result)
