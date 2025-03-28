import autogen
import yaml
import sys
from logger import Logger
from utils import *
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, register_function
import json
from tqdm import tqdm
from RAG.rag_utils import create_collection, upload_txt_file, rag_search
from functions import function_map


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
        return json.loads(data)


if __name__ == "__main__":

    sys.stdout = Logger()   
    config = load_yaml("config.yaml")
    config_list_zhipu = load_ZhiPu(config)
    data = load_data("/workspace/Projects/EDU/data/bio/bio.json")

    # Create a user proxy agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",  # This will terminate the conversation when needed
        max_consecutive_auto_reply=50,  # Don't auto-reply by default
        code_execution_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        llm_config=False
    )

    reinforce_agent = AssistantAgent(
        name="reinforce_agent",
        system_message="你是一个专业的老师，精通于根据学生薄弱点使用rag_search获得题库后出一套训练习题。根据薄弱点需要使用rag_search将学生薄弱点进行搜索，找到相关问题，并给学生输出一份针对其薄弱点的题目试卷。知识库名字为bio_kb。",
        llm_config={"config_list": config_list_zhipu, "seed": 42},
        max_consecutive_auto_reply=30
    )

    summary_agent = AssistantAgent(
        name="summary_agent",
        system_message="你是一个专业的分析学生错题专家.可以总结并概括学生薄弱点。",
        llm_config={"config_list": config_list_zhipu, "seed": 42},
        max_consecutive_auto_reply=30
    )


    # Register the calculator function to the two agents.
    register_function(
        rag_search,
        caller=reinforce_agent,  # The assistant agent can suggest calls to the calculator.
        executor=user_proxy,  # The user proxy agent can execute the calculator calls.
        name="rag_search",  # By default, the function name is used as the tool name.
        description="Search for bio knowledge from knowledge base",  # A description of the tool.
    )

    def state_transition(last_speaker, groupchat):
        messages = groupchat.messages

        if last_speaker is chat_manager:
            # init -> retrieve
            return reinforce_agent

        if last_speaker is summary_agent:
            # init -> retrieve
            return reinforce_agent
        elif last_speaker is reinforce_agent:
            # retrieve: action 1 -> action 2
            return user_proxy
        elif last_speaker is user_proxy:

            return None





    groupchat = autogen.GroupChat(
    agents=[summary_agent, reinforce_agent],
    messages=[],
    max_round=20,
    speaker_selection_method="auto"  # Add this line
    )


    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list_zhipu, "seed": 42})
    # 在程序结束时关闭 log 文件
    user_proxy.initiate_chat(
    manager,
    message=f"请分析学生的错题{data}"
)
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal # 恢复默认输出





