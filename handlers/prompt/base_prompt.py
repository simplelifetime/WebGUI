from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasePromptTemplate(ABC):
    """
    提示模板的抽象基类，定义了所有提示模板必须实现的接口
    """
    
    @property
    @abstractmethod
    def SYSTEM_PROMPT(self) -> str:
        """系统提示内容"""
        pass
    
    @property
    @abstractmethod
    def USER_PROMPT_TEMPLATE(self) -> str:
        """用户提示模板"""
        pass
    
    @abstractmethod
    def create_messages(self, task: str, history_responses: List[str], 
                        history_images: List[str], current_screenshot: str, 
                        history_n: int = 5) -> List[Dict[str, Any]]:
        """
        创建LLM消息列表
        
        参数:
            task: 任务描述
            history_responses: 历史响应列表
            history_images: 历史图像列表
            current_screenshot: 当前截图
            history_n: 包含的历史记录数量
            
        返回:
            消息列表
        """
        pass 