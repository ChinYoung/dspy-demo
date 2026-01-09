# glm4_dspy.py

import os
import logging
from typing import List, Dict, Optional

import dspy
from zai import ZhipuAiClient

logging.basicConfig(level=logging.INFO)


class LmGlm(dspy.LM):
    """
    DSPy 适配器 for 智谱 AI GLM-4 系列模型(如 glm-4, glm-4-plus, glm-4-air, glm-4-flash)
    """

    def __init__(
        self,
        model: str = "glm-4.7",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        timeout: int = 60,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.model_name = model
        self.api_key = api_key or os.getenv("ZAI_API_KEY")
        if not self.api_key:
            raise ValueError("ZAI_API_KEY 未设置，请通过参数或环境变量提供。")

        self.client = ZhipuAiClient(api_key=os.getenv("ZAI_API_KEY"))
        self.timeout = timeout

        # 默认生成参数
        self.default_kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            **kwargs,
        }

        self.history: List[Dict] = []

    def basic_request(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        **kwargs,
    ):
        """
        向 GLM API 发送请求。
        支持传入单轮 prompt 字符串(prompt)或已构造的 messages 列表(messages)。
        """
        combined_kwargs = {**self.default_kwargs, **kwargs}

        # 如果提供了 messages，直接使用；否则将 prompt 封装为 user 消息
        if messages is None:
            if prompt is None:
                raise ValueError("Either 'prompt' or 'messages' must be provided.")
            messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=self.timeout,
                **combined_kwargs,
            )
            logging.info(f"GLM API 调用成功, prompt: {prompt} 响应: {response}")
            return response
        except Exception as e:
            logging.error(f"GLM API 调用失败: {e}")
            raise RuntimeError(f"GLM API 调用错误: {str(e)}") from e

    def __call__(self, prompt=None, messages=None, **kwargs) -> List[str]:
        """
        返回生成的文本列表(与 DSPy 接口对齐)
        支持与 BaseLM 一致的签名:prompt 或 messages。
        """
        response = self.basic_request(prompt=prompt, messages=messages, **kwargs)
        outputs = []
        for choice in getattr(response, "choices", []) or []:
            # try new style
            content = None
            if (
                hasattr(choice, "message")
                and getattr(choice.message, "content", None) is not None
            ):
                content = choice.message.content
            elif isinstance(choice, dict):
                # dict style
                content = choice.get("message", {}).get("content") or choice.get("text")
            else:
                # object attribute style
                content = getattr(choice, "text", None)
            if content is None:
                continue
            outputs.append(content.strip())
        return outputs

    @property
    def name(self):
        return self.model_name
