import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List
from WebGUI.custom_types import Message
from .base_handler import BaseLLMHandler

class Qwen2_5Handler(BaseLLMHandler):
    """Qwen2-VL模型的处理器实现"""
    
    def __init__(self, model_path: str, **kwargs):
        """
        初始化Qwen2-VL模型
        
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
        self.llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, 
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        
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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 准备输入
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # 生成输出
        with torch.inference_mode():
            generated_ids = self.llm.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 解码输出
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0] 