import autogen
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import AssistantAgent, config_list_from_json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
import logging
import traceback

# OpenAI LLM配置 - 使用智谱AI
config_list = [
    {
        "model": "GLM-4-Flash",
        "api_key": "398f5e33f62c6a9dc7cd95987b14aae6.rKQve5A1VUNNHW4Z",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "price": [0.002, 0.004],
    }
]

model = SentenceTransformer('all-MiniLM-L6-v2')
    

client = QdrantClient(
    url="https://1125ff6b-5368-4bf2-baf1-2b1e36fbc8fd.us-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3MTA3MTIyfQ.dR4GBwXt0oPsXuFzCsSEYXf4A7PCDtofuEDXpwJ6_p8"
)

# 自定义 embedding 函数
def local_embedding_function(texts):
    return model.encode(texts).tolist()

# 自定义文档检索函数
def custom_retrieve_documents(query, top_k=5):
    try:
        # 生成查询嵌入
        query_embedding = model.encode(query).tolist()

        # 执行语义检索
        results = client.search(
            collection_name="autogen_docs", 
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        # 提取文档内容
        retrieved_docs = [
            result.payload.get('text', '') for result in results
        ]

        # 打印检索详情
        logger.debug(f"查询: {query}")
        for result in results:
            logger.debug(f"相关度: {result.score}")
            logger.debug(f"文档内容: {result.payload}")

        return retrieved_docs
    except Exception as e:
        logger.error(f"文档检索错误: {e}")
        logger.error(traceback.format_exc())
        return []

# 创建检索配置
retrieve_config = {
    "task": "qa",
    "chunk_token_size": 500,
    "model": config_list[0]["model"],
    "vector_db": "qdrant",
    "qdrant": {
        "url": "https://1125ff6b-5368-4bf2-baf1-2b1e36fbc8fd.us-west-2-0.aws.cloud.qdrant.io:6333",
        "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3MTA3MTIyfQ.dR4GBwXt0oPsXuFzCsSEYXf4A7PCDtofuEDXpwJ6_p8",
        "collection_name": "autogen_docs",
        "prefer_grpc": False
    },
    "embedding_function": local_embedding_function,
    "custom_retrieve_function": custom_retrieve_documents,
    # "embedding_model": "all-MiniLM-L6-v2", /
    "get_or_create": False,
    "overwrite": False,
    "docs_path": None,
    "top_k": 5,
    "verbose": True
}

# 创建普通助手代理
assistant = AssistantAgent(
    name="assistant",
    system_message="你是一个擅长回答问题的助手。使用检索到的文档回答用户的问题。",
    llm_config={"config_list": config_list}
)

# 创建检索用户代理
try:
    user_proxy = RetrieveUserProxyAgent(
        name="retrieve_user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        retrieve_config=retrieve_config,
        code_execution_config={"work_dir": "coding"}
    )

    # 开始RAG对话
    user_proxy.initiate_chat(
        assistant,
        message="请问小红喜欢的人是谁"
    )

except Exception as e:
    logger.error(f"对话初始化错误: {e}")
    logger.error(traceback.format_exc())