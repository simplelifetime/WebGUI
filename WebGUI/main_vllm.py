import asyncio
import argparse
import os
import io
import base64
from PIL import Image
from vllm import LLM, EngineArgs, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from screenshot_agent_vllm import ScreenshotAgentVLLM
from qwen_vl_utils import process_vision_info

# 设置vLLM环境变量以降低内存使用
os.environ["VLLM_USE_MODELOPT"] = "1"
os.environ["VLLM_MAX_MODEL_LEN"] = "4096"

async def main():
    parser = argparse.ArgumentParser(description="基于VLLM推理的GUI自动化代理")
    parser.add_argument("--task", type=str, default="请搜索姚明的图片，找出他穿着11号球衣的照片",
                       help="要执行的任务描述")
    parser.add_argument("--llm", type=str, choices=["UITARS", "Qwen2", "Qwen2_5", "Qwen2_5_R1"], default="Qwen2_5_R1",
                       help="要使用的LLM类型")
    parser.add_argument("--max_steps", type=int, default=10,
                       help="最大执行步数")
    parser.add_argument("--start_url", type=str, default="https://www.baidu.com/",
                       help="起始URL")
    parser.add_argument("--gpu_ids", type=str, default=None,
                      help="指定使用的GPU ID，例如 '0,1'")
    
    args = parser.parse_args()
    
    # 如果指定了GPU，设置环境变量
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # 创建代理实例
    agent = ScreenshotAgentVLLM(
        task=args.task,
        llm_type=args.llm,
        max_steps=args.max_steps,
        start_url=args.start_url
    )
    
    # vLLM 配置
    model_path = '/mnt/data/zkliu/hf_models/qwen2_5-vl-7b-instruct/'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 使用EngineArgs明确设置GPU内存分配
    engine_args = EngineArgs(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=16392,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 10}
    )
    
    # 初始化vLLM引擎
    print("start initializing vllm_engine\n############\n")
    vllm_engine = LLM(**engine_args.__dict__)
    print("vllm_engine initialized\n############\n")
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.8, 
        repetition_penalty=1.05, 
        max_tokens=512
    )
    
    # 定义一个同步的推理函数
    def vllm_infer_sync(messages):
        # 提取并处理消息
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(f"prompt_length: {len(tokenizer(prompt)['input_ids'])}")
        image_data, _ = process_vision_info(messages)
        
        # 准备输入
        vllm_inputs = [{'prompt': prompt, "multi_modal_data": {'image': image_data}}]
        
        # 生成响应
        outputs = vllm_engine.generate(vllm_inputs, sampling_params)
        
        # 提取生成的文本
        if outputs and outputs[0].outputs:
            return outputs[0].outputs[0].text
        else:
            return "<think>\n无法获取有效响应\n</think>\n<action>\nwait()\n</action>"
    
    # 包装为异步函数
    async def vllm_infer(messages):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, vllm_infer_sync, messages)
    
    try:
        # 初始化并获取首次消息
        initial_messages = await agent.setup()
        
        # 第一次调用vLLM推理
        vllm_response = await vllm_infer(initial_messages)
        
        # 交互循环
        completed = False
        while not completed:
            # 处理vLLM响应
            next_messages, completed, answer = await agent.process_response(vllm_response)
            
            # 如果任务已完成，退出循环
            if completed:
                if answer:
                    print(f"任务完成，答案: {answer}")
                break
                
            # 否则继续调用vLLM推理
            vllm_response = await vllm_infer(next_messages)
            
            print(f"vllm_response: {vllm_response}")
            
    finally:
        # 确保资源被清理
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 