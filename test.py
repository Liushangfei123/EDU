import autogen
from langchain_google_vertexai import VertexAI
from google.auth import default
import os

# --------------------- 配置 Google Cloud 凭证 ---------------------
try:
    credentials, project_id = default()
    if not project_id:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError(
                "Could not determine Google Cloud project ID. "
                "Ensure GOOGLE_CLOUD_PROJECT environment variable is set or "
                "you are running in an environment with default credentials."
            )
    print(f"Using Google Cloud Project: {project_id}")
except Exception as e:
    print(f"Error loading Google Cloud credentials: {e}")
    print(
        "Please ensure your environment is correctly configured for Google Cloud authentication."
    )
    exit()

# --------------------- 初始化 LangChain Gemini 模型 ---------------------
try:
    llm_gemini = VertexAI(model_name="gemini-pro", project=project_id, location="asia-east1")
    print("Successfully initialized LangChain Gemini model.")
except Exception as e:
    print(f"Error initializing LangChain VertexAI: {e}")
    exit()

# --------------------- 配置 AutoGen 使用 LangChain LLM ---------------------
config_list_langchain = [
    {
        "model": "gemini-pro",
        "api_key": "LANGCHAIN_LLM",  # 特殊标记，告诉 AutoGen 使用 LangChain LLM
    }
]

# --------------------- 创建 UserProxyAgent ---------------------
user_proxy = autogen.UserProxyAgent(
    name="用户",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "coding"},
)

# --------------------- 创建 AssistantAgent，并使用 LangChain LLM ---------------------
assistant_gemini_langchain = autogen.AssistantAgent(
    name="Gemini_助手",
    llm_config={"config_list": config_list_langchain, "functions": None},  # functions 设为 None
    system_message="""你是一个使用 Google Gemini Pro 模型 (通过 LangChain) 提供支持的助手。
    你擅长回答各种问题，请清晰、简洁地回复。""",
)

# --------------------- 设置 AutoGen 使用 LangChain LLM 实例 ---------------------
assistant_gemini_langchain.llm = llm_gemini

# --------------------- 启动聊天 ---------------------
initial_message = "你好！请简单介绍一下 Google Gemini Pro 模型。"

try:
    user_proxy.initiate_chat(
        assistant_gemini_langchain,
        message=initial_message,
    )
except Exception as e:
    print(f"Error during chat initiation: {e}")

# --------------------- 进一步交互示例 (可选) ---------------------
# if user_proxy.human_input_mode != "ALWAYS":
#     user_response = "很有趣，谢谢！你还能做些什么？"
#     try:
#         user_proxy.send(
#             message=user_response,
#             recipient=assistant_gemini_langchain,
#         )
#     except Exception as e:
#         print(f"Error during subsequent interaction: {e}")