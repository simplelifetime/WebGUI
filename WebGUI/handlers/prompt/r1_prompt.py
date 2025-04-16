from typing import List, Dict, Any
from .base_prompt import BasePromptTemplate
from WebGUI.custom_types import Message

class R1PromptTemplate(BasePromptTemplate):
    """默认的提示模板实现"""
    
    @property
    def SYSTEM_PROMPT(self) -> str:
        return """You are Qwen, a helpful assistant."""

    @property
    def USER_PROMPT_TEMPLATE(self) -> str:
        return """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. You need to think first, then perform the action.

## Output Format
Your response MUST follow this exact format:

```
<think>
Your thought for the next step here.
</think>
<action>
Your action here.
</action>
```

If the task is completed, add a summary answer at the end:

```
<think>
Your thought for the next step here.
</think>
<action>
Your action here.
</action>
<answer>
Your answerh here.
</answer>
```

## Output Examples

Example 1 (During task):
```
<think>
I need to search for the search box first and then type the query.
</think>
<action>
click(start_box='(100,100)')
type(content='openai')
</action>
```

Example 2 (Task completed):
```
<think>
I have successfully completed the search and found the required information.
</think>
<action>
finished()
</action>
<answer>
1941
</answer>
```

## Action Space

click(start_box='(x1,y1)')
left_double(start_box='(x1,y1)')
right_single(start_box='(x1,y1)')
drag(start_box='(x1,y1)', end_box='(x2,y2)')
hotkey(key='')
type(content='')
scroll(start_box='(x1,y1)', direction='down or up or right or left')
wait()
finished()
call_user()
goto(url='') 
goback()


## Action Explaination

click: Click at the specified coordinates (x1,y1) with the left mouse button
left_double: Double click at the specified coordinates (x1,y1) with the left mouse button
right_single: Click at the specified coordinates (x1,y1) with the right mouse button
drag: Click and hold at start coordinates (x1,y1), then drag to end coordinates (x2,y2)
hotkey: Press the specified keyboard key combination (e.g., 'Ctrl C' for copy)
type: Type the specified text content. Use '\\n' at the end to submit the input
scroll: Scroll in the specified direction (up/down/left/right) at the given coordinates
wait: Pause for 5 seconds and take a screenshot to check for changes
finished: Indicate that the task has been completed successfully
call_user: Request user assistance when the task is unsolvable or help is needed
goto: Navigate to a specific URL in the browser
goback: Return to the previous page in browser history


## Notes
1. You MUST wrap your thoughts in <think> tags
2. You MUST wrap your actions in <action> tags
3. When task is completed, you MUST add a summary in <answer> tags. If the task requires a accurate answer, provide an answer in wrapped in <answer> tags. If not, just add a summary in <answer> tags.
4. Use English in all sections
5. Your coordinate should be in the format of `(x,y)`. (0,0) is the top-left corner of the screen.
6. If you have multiple actions at one time, split them with a empty line.
7. Thinking is crucial - ALWAYS think carefully before taking any action. If you can solve a problem through careful analysis and thinking, prioritize that over immediate actions.
8. When typing search queries, think carefully about the most effective and precise query that will yield the fastest results. Avoid random or vague searches - optimize your queries for accuracy and efficiency.
9. If a page fails to load or doesn't show relevant results, consider using goback() to return to the previous page or goto() to navigate to a specific page (including the initial search page) to try a different approach.
10. If you have used scroll more than three times in a row, you should consider using goback() or goto() to change the page, since you may be stuck in a loop.

We will keep all the history of your thought and action traces (In text modal). You can use the history to improve your performance. Also, we only keep the last {history_n} screenshots for your reference.

## User Instruction
{task}

## Action History
"""

    def create_messages(self, task: str, history_responses: List[str], 
                       history_images: List[str], current_screenshot: str, 
                       history_n: int = 8) -> List[Message]:
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
                "content": self.USER_PROMPT_TEMPLATE.format(task=task, history_n=history_n)
            }
        ]
        
        # 添加最近的历史记录
        if len(history_responses) > 0:
            history_start = max(0, len(history_responses) - history_n)
            for history_idx, history_response in enumerate(history_responses):
                if history_idx >= history_start:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{history_images[history_idx]}"
                            }
                        ]
                    })
                    messages.append({
                    "role": "assistant",
                    "content": history_response
                })
                    
                else:
                    messages[1]["content"] += f"\nStep {history_idx + 1}:\n {history_response}"
                    
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
