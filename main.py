import autogen
import yaml
import sys


class Logger(object):
    def __init__(self, filename="autogen.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

sys.stdout = Logger()


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


zhipu_config = config["client"]["zhipu"]
# config_list_gpt = [client_config]

config_list_gpt = [{**zhipu_config}]

# User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="你是一个用户代理，负责接收用户的需求并协调整个流程。你需要清晰地理解用户的需求，并将任务分配给合适的 Agent。",
    code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
    human_input_mode="ALWAYS",  # 允许用户在每个步骤进行干预
)

# Market Analyst Agent
market_analyst = autogen.AssistantAgent(
    name="Market_Analyst",
    llm_config={"config_list": config_list_gpt, "seed": 42},
    system_message="你是一位资深的市场分析师。你的任务是根据用户需求进行市场调研，分析目标用户、竞争对手和市场趋势。你需要使用搜索引擎等工具来获取信息，并总结市场机会和潜在风险。",
)

# Product Manager Agent
product_manager = autogen.AssistantAgent(
    name="Product_Manager",
    llm_config={"config_list": config_list_gpt, "seed": 42},
    system_message="你是一位经验丰富的产品经理。你的任务是根据市场分析结果，生成具体的产品概念和初步的产品规格。你需要考虑用户痛点、市场机会和技术可行性。",
)

# Technical Lead Agent
technical_lead = autogen.AssistantAgent(
    name="Technical_Lead",
    llm_config={"config_list": config_list_gpt, "seed": 42},
    system_message="你是一位资深的技术主管。你的任务是评估产品概念的技术可行性，识别关键技术和潜在的技术风险。你需要考虑现有技术、开发成本和时间等因素。",
)

# Report Writer Agent
report_writer = autogen.AssistantAgent(
    name="Report_Writer",
    llm_config={"config_list": config_list_gpt, "seed": 42},
    system_message="你是一位专业的报告撰写员。你的任务是整合市场分析师、产品经理和技术主管的输出，生成一份包含市场分析、产品概念和技术可行性评估的初步报告。",
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, market_analyst, product_manager, technical_lead, report_writer],
    messages=[],
    max_round=20,  # 设置最大对话轮数
)
manager = autogen.GroupChatManager(llm_config={"config_list": config_list_gpt, "seed": 42}, groupchat=groupchat)

user_proxy.initiate_chat(
    manager,
    message="我想开发一款能够提升用户生活品质的智能家居产品。请进行市场调研，生成产品概念，并评估其初步的技术可行性，最终生成一份报告。",
)


# 在程序结束时关闭 log 文件
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal # 恢复默认输出