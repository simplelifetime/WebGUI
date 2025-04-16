import asyncio
from typing import List, Dict, Optional, Tuple, Any
from WebGUI.custom_types import FINISH_WORD, WAIT_WORD, ERROR_WORD
from WebGUI.actions import ActionParser, R1ActionParser
from WebGUI.browser_controller import BrowserController

# LLM配置
LLM_CONFIGS = {
    "UITARS": {
        "prompt_class": "DefaultPromptTemplate",
        "coord_system": "relative",
        "action_parser": ActionParser
    },
    "Qwen2": {
        "prompt_class": "Gemma3PromptTemplate",
        "coord_system": "absolute",
        "action_parser": ActionParser
    },
    "Qwen2_5": {
        "prompt_class": "Gemma3PromptTemplate",
        "coord_system": "absolute",
        "action_parser": ActionParser
    },
    "Qwen2_5_R1": {
        "prompt_class": "R1PromptTemplate",
        "coord_system": "absolute",
        "action_parser": R1ActionParser
    }
}

class ScreenshotAgentVLLM:
    def __init__(self, task: str, llm_type: str = "UITARS", max_steps: int = 15, start_url: str = "https://www.baidu.com/", history_n=5):
        self.task = task
        self.max_steps = max_steps
        self.start_url = start_url
        self.current_step = 0
        
        # 导入需要的模块
        from WebGUI.handlers.prompt import DefaultPromptTemplate, Gemma3PromptTemplate, R1PromptTemplate
        
        # 将字符串类名转换为实际类对象的映射
        prompt_classes = {
            "DefaultPromptTemplate": DefaultPromptTemplate,
            "Gemma3PromptTemplate": Gemma3PromptTemplate,
            "R1PromptTemplate": R1PromptTemplate
        }
        
        # 根据llm_type选择对应的LLM和提示模板
        if llm_type not in LLM_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}，可选值: {list(LLM_CONFIGS.keys())}")
        
        llm_config = LLM_CONFIGS[llm_type]
        
        # 将字符串类名转换为实际类对象
        prompt_class = prompt_classes[llm_config["prompt_class"]]
        
        self.prompt_template = prompt_class()
        self.action_parser = llm_config["action_parser"]
        
        # 初始化浏览器控制器，不启用日志
        self.browser_controller = BrowserController(enable_logging=False, coordinate_system=llm_config['coord_system'])
        
        self.history = []
        self.history_responses = []
        self.history_images = []
        self.history_n = history_n
        self.messages = []
        
        # 任务完成标志
        self.task_completed = False
        self.task_answer = None

    async def setup(self):
        """初始化浏览器并导航到起始页面"""
        await self.browser_controller.setup()
        await self.browser_controller.page.goto(self.start_url, timeout=10000)
        
        # 获取初始截图
        normal_screenshot, _ = await self.browser_controller.get_screenshots()
        self.history_images.append(normal_screenshot)
        
        # 创建初始消息
        return self.prompt_template.create_messages(
            self.task,
            self.history_responses,
            self.history_images,
            normal_screenshot,
            self.history_n
        )

    async def cleanup(self):
        """关闭浏览器和清理资源"""
        try:
            if self.browser_controller:
                await self.browser_controller.close()
                self.browser_controller = None
        except Exception as e:
            print(f"Warning: Error during browser cleanup: {e}")
        finally:
            # 确保所有资源都被清理
            self.history = []
            self.history_responses = []
            self.history_images = []
            self.messages = []
            self.current_step = 0
            self.task_completed = False
            self.task_answer = None

    async def process_response(self, response: str) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
        """
        处理外部VLLM返回的response，执行操作并返回新的消息
        
        参数:
            response: 外部VLLM的响应文本
            
        返回:
            Tuple包含:
            - 下一次推理的消息
            - 任务是否完成的标志
            - 如果任务完成，包含最终答案；否则为None
        """
        # 存储响应
        self.history_responses.append(response)
        self.current_step += 1
        
        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            # print(f"Reached maximum steps limit ({self.max_steps})")
            # await self.cleanup()
            return [], True, None
        
        # 解析响应并执行操作
        try:
            if self.action_parser == R1ActionParser:
                actions, answer = self.action_parser.parse_llm_response(response)
                if answer:
                    # print(f"\nTask completed with answer: {answer}")
                    self.task_completed = True
                    self.task_answer = answer
                    # await self.cleanup()
                    return [], True, answer
            else:
                actions = self.action_parser.parse_llm_response(response)
            
            # 执行操作
            for action in actions:
                # print(f"\nExecuting action: {action}")
                
                result = await self.browser_controller.execute_action(action)
                self.history.append(action)
                
                if result == FINISH_WORD:
                    print("\nTask completed!")
                    self.task_completed = True
                    # await self.cleanup()
                    return [], True, None
                elif result == ERROR_WORD:
                    print("\nError occurred, stopping execution.")
                    # await self.cleanup()
                    return [], True, None
                elif result == WAIT_WORD:
                    await asyncio.sleep(2)
                else:
                    await asyncio.sleep(1)
            
            # 获取新的截图和创建新的消息
            normal_screenshot, _ = await self.browser_controller.get_screenshots()
            self.history_images.append(normal_screenshot)
            
            messages = self.prompt_template.create_messages(
                self.task,
                self.history_responses,
                self.history_images,
                normal_screenshot,
                self.history_n
            )
            self.messages = messages
            
            return messages, False, None
            
        except Exception as e:
            # print(f"Error in action execution: {e}")
            # await self.cleanup()
            return [], True, None