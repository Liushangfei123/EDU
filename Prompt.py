from langchain.prompts import PromptTemplate

# 定义错误分析与对比建议的链
analysis_prompt = PromptTemplate(
    input_variables=["question", "wrong_answer"],
    template="""
你是一名优秀的教学助理。请仔细阅读以下题目和学生的错误答案，
并详细分析错误答案中存在的问题以及学生可能的误解。

题目: {question}
学生的错误答案: {wrong_answer}

请列出错误点、分析错误原因，并说明你的思考过程。
    """
)

advice_prompt = PromptTemplate(
    input_variables=["analysis", "correct_answer"],
    template="""
根据以下分析:
{analysis}

正确答案是: {correct_answer}

请详细说明学生错误的根本原因，并给出具体的改进建议，
帮助学生理解正确答案中的关键点。
    """
)

# 在 Prompt.py 中添加
summary_prompt = PromptTemplate.from_template(
    "以下是一系列问题的错误回答、正确回答和分析建议：\n\n{mistakes_summary}\n\n"
    "请对这些错题进行总结，分析学生普遍存在的问题，并给出整体的改进建议。"
)