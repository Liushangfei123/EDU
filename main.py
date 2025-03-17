
import json
from langchain.schema.runnable import RunnableSequence, RunnableLambda
from ZHIPU import ChatGLM
from Prompt import analysis_prompt, advice_prompt, summary_prompt
from tqdm import tqdm

zhipu_llm = ChatGLM()


# 处理链
analysis_chain = analysis_prompt | zhipu_llm
advice_chain = advice_prompt | zhipu_llm
summary_chain = summary_prompt | zhipu_llm




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

# 处理多个问题的函数
def process_multiple_questions(questions):
    results = []
    for question in tqdm(questions):
        result = react_chain.invoke(question)
        results.append({
            "question": question["question"],
            "wrong_answer": question["wrong_answer"],
            "correct_answer": question["correct_answer"],
            "advice": result
        })
    return results

# 生成错题总结的函数
def generate_summary(results):
    summary_input = "\n".join([
        f"问题：{r['question']}\n"
        f"错误回答：{r['wrong_answer']}\n"
        f"正确回答：{r['correct_answer']}\n"
        f"分析建议：{r['advice']}\n"
        for r in results
    ])
    return summary_chain.invoke({"mistakes_summary": summary_input})




# 示例使用
if __name__ == "__main__":
    with open("data/bio.json", 'r', encoding='utf-8') as file:
        data = file.read()
    # Parse the JSON data
    data = json.loads(data)
    results = process_multiple_questions(data)
    save_path = "result.txt"
    # 保存结果到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(generate_summary(results))

    # print("【改进建议】\n", result)  # 第二个输出是建议/