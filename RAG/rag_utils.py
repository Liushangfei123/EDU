from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import json
import time
from zhipuai import ZhipuAI
from typing import List, Dict, Any

from keys import *

def create_collection(collection_name:str,dimension:int =1024):
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


def generate_embeddings(texts:str)-> List[float]:
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

def upload_txt_file(text, COLLECTION_NAME, chunk_size=100, chunk_overlap=50):

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

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

def rag_search(collection_name, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    query_vector = generate_embeddings(query)[0]
    try:
        response = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        results = []
        for result in response:
            results.append({
                "text": result.payload.get('text', ''),
                "similarity": result.score,
                "doc_id": result.payload.get("doc_id", ""),})

        return results

    except Exception as e:
        print(f"检索失败: {e}")
        return None




if __name__ == '__main__':

    file_path = "../data/bio/kb.txt"
    COLLECTION_NAME = "bio_kb"
    create_collection(COLLECTION_NAME, 1024)
    with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    upload_txt_file(text, COLLECTION_NAME)
    query = "什么是达尔文"
    print(rag_search(collection_name=COLLECTION_NAME, query=query))