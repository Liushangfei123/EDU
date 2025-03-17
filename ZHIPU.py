from langchain.llms.base import LLM
from typing import ClassVar
import requests


class ChatGLM(LLM):
    api_key: ClassVar[str] = "398f5e33f62c6a9dc7cd95987b14aae6.rKQve5A1VUNNHW4Z"  # 替换为你的智谱 API 密钥
    url: ClassVar[str] = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def _call(self, prompt: str, stop=None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "GLM-4-Flash",
            "messages": [{"role": "system", "content": "你是一个智能助手"}, {"role": "user", "content": prompt}],
            "max_tokens": 1000
        }
        response = requests.post(self.url, json=payload, headers=headers)
        return response.json()["choices"][0]["message"]["content"] if response.status_code == 200 else f"Error: {response.text}"

    @property
    def _llm_type(self):
        return "chatglm"

    @property  
    def _identifying_params(self):
        return {"model": "GLM-4-Flash"}