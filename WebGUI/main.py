import asyncio
import argparse
from screenshot_agent import ScreenshotAgent

async def main():
    parser = argparse.ArgumentParser(description="基于大模型的GUI自动化代理")
    parser.add_argument("--task", type=str, default="请搜索姚明的图片，找出他穿着11号球衣的照片",
                       help="要执行的任务描述")
    parser.add_argument("--llm", type=str, choices=["UITARS", "Qwen2", "Qwen2_5", "Qwen2_5_R1"], default="UITARS",
                       help="要使用的LLM类型")
    parser.add_argument("--max_steps", type=int, default=10,
                       help="最大执行步数")
    parser.add_argument("--start_url", type=str, default="https://www.google.com/",
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