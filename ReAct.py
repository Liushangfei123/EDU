

from langchain.schema.runnable import RunnableSequence, RunnableLambda
from ZHIPU import ChatGLM
from Prompt import analysis_prompt, advice_prompt

zhipu_llm = ChatGLM()


# 处理链
analysis_chain = analysis_prompt | zhipu_llm
advice_chain = advice_prompt | zhipu_llm

# 组合链
def format_input(inputs):
    analysis_output = inputs["analysis_output"]
    return {"analysis": analysis_output, "correct_answer": inputs["correct_answer"]}

react_chain = {
    "question": lambda x: x["question"],
    "wrong_answer": lambda x: x["wrong_answer"],
    "correct_answer": lambda x: x["correct_answer"],
    "analysis_output": analysis_chain
} | RunnableLambda(format_input) | advice_chain

# 示例使用
if __name__ == "__main__":
    question = "请证明勾股定理。"
    wrong_answer = "学生在证明过程中错误地利用了圆的性质，导致证明不严谨。"
    correct_answer = "通过构造辅助线证明三角形的相似性，从而推导出 a^2 + b^2 = c^2。"

    result = react_chain.invoke({"question": question, "wrong_answer": wrong_answer, "correct_answer": correct_answer})

    print("【改进建议】\n", result)  # 第二个输出是建议