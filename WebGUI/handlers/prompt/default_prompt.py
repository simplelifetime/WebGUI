from typing import List, Dict, Any
from .base_prompt import BasePromptTemplate
from WebGUI.custom_types import Message

class DefaultPromptTemplate(BasePromptTemplate):
    """默认的提示模板实现"""
    
    @property
    def SYSTEM_PROMPT(self) -> str:
        return """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task."""

    @property
    def USER_PROMPT_TEMPLATE(self) -> str:
        return """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='(x1,y1)')
left_double(start_box='(x1,y1)')
right_single(start_box='(x1,y1)')
drag(start_box='(x1,y1)', end_box='(x2,y2)')
hotkey(key='')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(start_box='(x1,y1)', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


## Note
- Use English in `Thought` part.
- Summarize your next action (with its target element) in `Thought` part.

## User Instruction
{task}
"""

    def create_messages(self, task: str, history_responses: List[str], 
                       history_images: List[str], current_screenshot: str, 
                       history_n: int = 5) -> List[Message]:
        """
        创建LLM消息列表，包括历史记录和当前截图
        
        参数:
            task: 任务描述
            history_responses: 历史响应列表
            history_images: 历史图像列表
            current_screenshot: 当前截图
            history_n: 包含的历史记录数量
            
        返回:
            消息列表
        """
        messages: List[Message] = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": self.USER_PROMPT_TEMPLATE.format(task=task)
            }
        ]
        
        # 添加最近的历史记录
        if len(history_responses) > 0:
            history_start = max(0, len(history_responses) - history_n)
            for history_idx, history_response in enumerate(history_responses[history_start:]):
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{history_images[history_start + history_idx]}"
                        }
                    ]
                })
                messages.append({
                    "role": "assistant",
                    "content": history_response
                })
        
        # 添加当前截图
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{current_screenshot}"
                }
            ]
        })
        
        return messages
