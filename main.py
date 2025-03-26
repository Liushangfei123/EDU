import autogen
import yaml
import sys
from logger import Logger
from utils import *
from autogen import AssistantAgent
import json
from tqdm import tqdm
from RAG.rag_utils import create_collection, upload_txt_file, rag_search

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
        return json.loads(data)


if __name__ == "__main__":

    sys.stdout = Logger()   
    config = load_yaml("config.yaml")
    config_list_zhipu = load_ZhiPu(config)
    data = load_data("data/bio.json")

    analysis_agent = AssistantAgent(
        name="analysis_agent",
        system_message="你是一个专业的生物老师，你需要仔细阅读学生错题，理解学生的错误原因。分析学生是以下几点中，哪几点比较薄弱。知识点：**细胞** 涵盖细胞的基本结构、植物细胞和动物细胞的区别、细胞的生命活动等。**生态系统与环境** 包括生态系统组成、生物与环境关系、生态平衡和生物多样性。**人体生理系统** 研究人体各个系统，如消化系统、呼吸系统、循环系统、神经系统等。**遗传与进化** 介绍基本遗传规律、基因概念、遗传变异和生物进化理论。植物生理 探讨光合作用、植物生长发育、器官结构和对环境的响应。你的回答结果如下：薄弱点：{**具体薄弱点**}，你的输出不能有任何解释。",
        llm_config={"config_list": config_list_zhipu, "seed": 42},
        max_consecutive_auto_reply=2
    )

    reinforce_agent = AssistantAgent(
        name="reinforce_agent",
        system_message="你是一个专业的分析学生错题的强化者，你需要将学生的错题进行强化，帮助学生理解错题。你需要使用中文进行交流。",
        llm_config={"config_list": config_list_zhipu, "seed": 42},
        function_map={"rag_search": rag_search},
        max_consecutive_auto_reply=1
    )



    # 在程序结束时关闭 log 文件
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal # 恢复默认输出





