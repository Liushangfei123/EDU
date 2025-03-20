import autogen
import yaml
import sys
from logger import Logger
from Agents import *
from autogen import AssistantAgent
import json
from tqdm import tqdm


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
        return json.loads(data)


if __name__ == "__main__":

    sys.stdout = Logger()   
    config = load_yaml("config.yaml")
    config_list_zhipu = load_ZhiPu(config)
    config_list_gemini = load_Gemini(config)
    config_list_deepseeker = load_deepseeker(config)
    data = load_data("data/bio.json")

    analysis_agent = AssistantAgent(
        name="analysis_agent",
        system_message="你是一个专业的错题分析老师，你需要仔细阅读学生错题，理解学生的错误原因，并给出改进建议。你需要使用中文进行交流。",
        llm_config={"config_list": config_list_zhipu, "seed": 42},
        max_consecutive_auto_reply=2
    )

    summary_agent = AssistantAgent(
        name="summary_agent",
        system_message="你是一个专业的分析学生错题的总结者，你需要将学生的错题进行总结，并给出总结报告。你需要使用中文进行交流。",
        llm_config={"config_list": config_list_gemini, "seed": 42},
        max_consecutive_auto_reply=1
    )
    analysis_results = []
    for question in tqdm(data):
        # 分析错题
        analysis_agent.initiate_chat(
            recipient=analysis_agent,
            message=question["question"],
        )
        analysis_result = analysis_agent.last_message()["content"]
        analysis_results.append(analysis_result)
    summary_message = "\n".join(analysis_results)

    # 薄弱点总结
    summary_agent.initiate_chat(
        recipient=summary_agent,
        message="请你根据错题分析老师的分析总结学生的薄弱点，总结该生接下来要学习哪些知识点。分析的结果如下：\n" + summary_message,
        max_consecutive_auto_reply=4
    )
    with open("data/summary.md", "w", encoding="utf-8") as f:
        f.write(summary_agent.last_message()["content"])

    # 输出总结报告
    print(summary_agent.last_message()["content"])


    # 在程序结束时关闭 log 文件
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal # 恢复默认输出





