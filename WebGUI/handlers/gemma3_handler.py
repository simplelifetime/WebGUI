import torch
from qwen_vl_utils import process_vision_info
from typing import List
from WebGUI.custom_types import Message
from .base_handler import BaseLLMHandler
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests


class Gemma3Handler(BaseLLMHandler):
    """Gemma3模型的处理器实现"""
    
    def __init__(self, model_path: str, **kwargs):
        """
        初始化Gemma3模型
        
        参数:
            model_path: 模型路径
            **kwargs: 额外的模型初始化参数
        """
        self.model_path = model_path
        
        # 获取选项或使用默认值
        torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        attn_implementation = kwargs.get('attn_implementation', "flash_attention_2")
        device_map = kwargs.get('device_map', "auto")
        
        # 初始化模型
        self.llm = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        ).eval()
        
        
        # 初始化处理器
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # 设置生成参数
        self.max_new_tokens = kwargs.get('max_new_tokens', 1024)
    
    def invoke(self, messages: List[Message]) -> str:
        """
        调用Qwen2-VL模型生成响应
        
        参数:
            messages: 消息列表，格式为custom_types.Message
            
        返回:
            模型生成的响应文本
        """
        # 处理聊天模板
        inputs = self.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        output_text = self.processor.decode(generation, skip_special_tokens=True)
        return output_text