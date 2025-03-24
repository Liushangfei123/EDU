from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import json
import time
from zhipuai import ZhipuAI

# 远程Qdrant连接配置
QDRANT_URL = "https://1125ff6b-5368-4bf2-baf1-2b1e36fbc8fd.us-west-2-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3MTA3MTIyfQ.dR4GBwXt0oPsXuFzCsSEYXf4A7PCDtofuEDXpwJ6_p8"
# 智谱AI API配置
ZHPIU_API_KEY = "398f5e33f62c6a9dc7cd95987b14aae6.rKQve5A1VUNNHW4Z"


def create_collection(collection_name,dimension=1024):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,  # ZhipuAI embedding-3的实际输出维度
                distance=models.Distance.COSINE
            )
        )
        print(f"创建了新集合 '{collection_name}'")
    except Exception as e:
        print(f"collection_exists: {e}")
        client.delete_collection(collection_name)
        print(f"集合 '{collection_name}' 删除成功。")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,  # ZhipuAI embedding-3的实际输出维度
                distance=models.Distance.COSINE
            )
        )
        print(f"创建了新集合 '{COLLECTION_NAME}'")


def generate_embeddings(texts):
    client_zhipu = ZhipuAI(api_key=ZHPIU_API_KEY)
    try:
        response = client_zhipu.embeddings.create(
            model="embedding-2",
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings
    except Exception as e:
        print(f"生成嵌入向量失败: {e}")
        return None

def upload_txt_file(file_path, COLLECTION_NAME, chunk_size=500, chunk_overlap=200):

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            text = file.read()

    print(f"成功读取文件: {file_path}")
    print(f"文本总长度: {len(text)} 字符")

    # 分割文本为块
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap) if len(text[i:i + chunk_size]) > 50]
    print(f"文本已分割为 {len(chunks)} 个块")

    # 生成嵌入向量
    embeddings = generate_embeddings(chunks)
    if embeddings is None:
        print("嵌入生成失败，退出。")
        return

    # 上传到Qdrant
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk, "source": file_path, "chunk_index": i}
        ) for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"上传批次 {i // batch_size + 1}/{(len(points) - 1) // batch_size + 1} 完成")

    print(f"成功上传文件 '{file_path}' 到Qdrant，共 {len(points)} 个文本块")


if __name__ == '__main__':

    file_path = "/workspace/Projects/EDU/data/romantic_data/story.txt"
    COLLECTION_NAME = "romantic_story"
    create_collection(COLLECTION_NAME, 1024)
    upload_txt_file(file_path, COLLECTION_NAME)