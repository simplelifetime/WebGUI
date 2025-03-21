import asyncio
import os
import base64
from dotenv import load_dotenv
from typing import List, Dict, Optional
from custom_types import FINISH_WORD, WAIT_WORD, ERROR_WORD
from actions import ActionParser
from browser_controller import BrowserController
import base64
from PIL import Image
import io
from handlers import Qwen2Handler, Qwen2_5Handler
from handlers.prompt import DefaultPromptTemplate, Gemma3PromptTemplate


load_dotenv()

# LLM配置
LLM_CONFIGS = {
    "UITARS": {
        "model_path": "/mnt/liuzikang/UI-TARS-7B-DPO",
        "handler_class": Qwen2Handler,
        "prompt_class": DefaultPromptTemplate
    },
    # "gemma": {
    #     "model_path": "/mnt/liuzikang/gemma-7b",  # 请更新为实际的Gemma模型路径
    #     "handler_class": Gemma3Handler,
    #     "prompt_class": Gemma3PromptTemplate
    # }
    "Qwen2_5": {
        "model_path": "/mnt/liuzikang/Qwen2.5-VL-7B-Instruct", 
        "handler_class": Qwen2_5Handler,
        "prompt_class": Gemma3PromptTemplate
    }
}

class ScreenshotAgent:
    def __init__(self, task: str, llm_type: str = "UITARS", max_steps: int = 15, start_url: str = "https://www.baidu.com/", enable_logging: bool = False):
        self.task = task
        self.max_steps = max_steps
        self.start_url = start_url
        
        # 根据llm_type选择对应的LLM和提示模板
        if llm_type not in LLM_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}，可选值: {list(LLM_CONFIGS.keys())}")
        
        llm_config = LLM_CONFIGS[llm_type]
        self.model_path = llm_config["model_path"]
        self.llm_handler = llm_config["handler_class"](self.model_path)
        self.prompt_template = llm_config["prompt_class"]()
        self.enable_logging = enable_logging
        
        self.browser_controller = BrowserController(self.enable_logging)
        
        self.history = []
        self.history_responses = []
        self.history_images = []
        self.history_images_toshow = []
        self.history_n = 5

    async def save_history_images(self):
        """Save history images to ./history_images folder"""
        os.makedirs("./history_images", exist_ok=True)
        
        # Clear all files in the directory
        for file in os.listdir("./history_images"):
            file_path = os.path.join("./history_images", file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        # Save new images
        all_images = []
        for idx, img_base64 in enumerate(self.history_images_toshow):
            img_data = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_data))
            all_images.append(img)
            with open(f"./history_images/step_{idx+1}.png", "wb") as f:
                f.write(img_data)
        
        if all_images:
            gif_path = f"./history_images/output.gif"
            all_images[0].save(
                gif_path,
                save_all=True,
                append_images=all_images[1:],
                duration=2000,  # 每帧显示时间，单位为毫秒
                loop=0  # 无限循环
            )
            print(f"GIF updated at {gif_path}")

    async def run(self):
        """Main execution loop"""
        await self.browser_controller.setup()
        print(f"Task: {self.task}")
        
        try:
            # Initial navigation
            await self.browser_controller.page.goto(self.start_url)
            
            while True:
                # Check max steps
                if len(self.history) >= self.max_steps:
                    print(f"\nReached maximum steps limit ({self.max_steps})")
                    break
                
                # Get screenshots
                normal_screenshot, cursor_screenshot = await self.browser_controller.get_screenshots()
                self.history_images.append(normal_screenshot)
                self.history_images_toshow.append(cursor_screenshot)
                
                # Create messages and get LLM response
                messages = self.prompt_template.create_messages(
                    self.task,
                    self.history_responses,
                    self.history_images,
                    normal_screenshot,
                    self.history_n
                )
                
                # await self.print_message(messages)
                response = self.llm_handler.invoke(messages)
                print(f"\nCurrent step: {len(self.history) + 1}")
                # print(f"Assistant:\n {response}")
                
                # Store response
                self.history_responses.append(response)
                
                # Parse and execute actions
                try:
                    actions = ActionParser.parse_llm_response(response)
                    
                    for action in actions:
                        print(f"\nExecuting action: {action}")
                        
                        result = await self.browser_controller.execute_action(action)
                        self.history.append(action)
                        
                        if result == FINISH_WORD:
                            print("\nTask completed!")
                            await self.save_history_images()
                            await self.browser_controller.close()
                            return
                        elif result == ERROR_WORD:
                            print("\nError occurred, stopping execution.")
                            await self.save_history_images()
                            await self.browser_controller.close()
                            return
                        elif result == WAIT_WORD:
                            await asyncio.sleep(2)
                        else:
                            await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"Error in action execution loop: {e}")
                    await self.save_history_images()
                    await self.browser_controller.close()
                    return
                    
        finally:
            await self.save_history_images()
            await self.browser_controller.close()
            
    async def print_message(self, messages):
        for mes in messages:
            if mes["role"] == "user":
                print(f"User:\n <image>")
            else:
                print(f"Assistant:\n {mes['content']}")
        print("--------------------------------\n")
            

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="基于大模型的GUI自动化代理")
    parser.add_argument("--task", type=str, default="请搜索姚明的图片，找出他穿着11号球衣的照片",
                       help="要执行的任务描述")
    parser.add_argument("--llm", type=str, choices=["UITARS", "gemma", "Qwen2_5"], default="UITARS",
                       help="要使用的LLM类型")
    parser.add_argument("--max_steps", type=int, default=10,
                       help="最大执行步数")
    parser.add_argument("--start_url", type=str, default="https://www.baidu.com/",
                       help="起始URL")
    parser.add_argument("--enable_logging", action="store_true",
                       help="是否启用日志记录")
    
    args = parser.parse_args()
    
    agent = ScreenshotAgent(
        task=args.task,
        llm_type=args.llm,
        max_steps=args.max_steps,
        start_url=args.start_url,
        enable_logging=args.enable_logging
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main()) 