from abc import ABC, abstractmethod
from typing import List, Any

class BaseLLMHandler(ABC):
    """
    基础LLM处理器抽象类，定义了所有LLM处理器必须实现的接口
    """
    
    @abstractmethod
    def __init__(self, model_path: str, **kwargs):
        """
        初始化模型
        
        参数:
            model_path: 模型路径
            **kwargs: 额外的模型初始化参数
        """
        pass
    
    @abstractmethod
    def invoke(self, messages: List[Any]) -> str:
        """
        调用LLM生成响应
        
        参数:
            messages: 消息列表，格式取决于具体的模型实现
            
        返回:
            模型生成的响应文本
        """
        pass 