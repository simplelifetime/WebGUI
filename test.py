import asyncio
import torch
import os
from transformers import AutoTokenizer, AutoProcessor
from tensordict import TensorDict

from qwen_vl_utils import process_vision_info
from WebGUI.screenshot_agent_vllm import ScreenshotAgentVLLM
from vllm import LLM, RequestOutput, SamplingParams
from typing import Dict, Any, Optional, Union, List, Tuple
import yaml
from vllm import LLM, EngineArgs, SamplingParams
import uuid
from datetime import datetime
import json
from tqdm import tqdm

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_env_vars():
    """设置分布式环境需要的环境变量"""
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"


class AgentState:
    """存储代理状态的类，便于管理多个代理实例"""
    def __init__(self, task_id, task, agent_id, start_url = 'https://www.google.com/', messages=None, task_history=None, level=0):
        self.task_id = task_id  # 原始任务ID
        self.task = task        # 任务描述
        self.agent_id = agent_id  # 代理ID (用于区分同一任务的不同采样)
        self.messages = messages or []  # 当前消息历史
        self.task_history = task_history or []  # 任务执行历史
        self.completed = False  # 任务是否完成
        self.answer = None      # 任务答案
        self.steps = 0          # 已执行步数
        self.unique_id = str(uuid.uuid4())[:8]  # 生成唯一ID用于跟踪
        self.agent = None       # 存储ScreenshotAgent实例
        self.output_dir = None  # 输出目录路径
        self.start_url = start_url  # 任务起始URL
        self.level = level
    
    def __str__(self):
        return f"Agent[{self.task_id}_{self.agent_id}] for '{self.task[:20]}...'"
    
    async def save_history(self):
        """保存代理的历史记录到JSON文件"""
        if self.agent and self.output_dir:
            # messages = self.agent.messages 
            images = self.agent.history_images
            print(f"saving images: {len(images)}")
            # messages.append({'role': 'assistant', 'content': self.task_history[-1]})
            # 保存完整的消息历史和任务历史
            saved_info = {
                'task_history': self.task_history,  # 保存任务历史
                'answer': self.answer,
                'completed': self.completed,
                'steps': self.steps,
                'images': images,
                'task': self.task,
                'start_url': self.start_url,
                'level': self.level
            }
            with open(f"{self.output_dir}/agent_info_{self.task_id}.json", 'w', encoding='utf-8') as f:
                json.dump(saved_info, f, ensure_ascii=False, indent=4)


async def test_generate_sequences_web():
    try:
        # 设置分布式环境变量
        setup_env_vars()
        
        # 创建统一的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"/mnt/data/zkliu/agent_tracs/agent_histories_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
        
        # 模型路径
        model_path = '/mnt/data/zkliu/hf_models/qwen2_5-vl-72b-instruct/'
        
        config = load_config('configs/config.yaml')
        rollout_config = config['worker']['rollout']
        actor_config = config['worker']['actor']
        data_config = config['data']
        
        # 提取vLLM相关配置
        tensor_parallel_size = rollout_config['tensor_parallel_size']
        tensor_parallel_size = 4
        gpu_memory_utilization = rollout_config['gpu_memory_utilization']
        gpu_memory_utilization = 0.85
        enforce_eager = rollout_config['enforce_eager']
        enable_chunked_prefill = rollout_config['enable_chunked_prefill']
        limit_images = rollout_config['limit_images']
        
        # 检查世界大小
        world_size = torch.cuda.device_count()
        if tensor_parallel_size > world_size:
            print(f"Warning: tensor_parallel_size ({tensor_parallel_size}) is greater than available devices ({world_size})")
            tensor_parallel_size = world_size
            print(f"Setting tensor_parallel_size to {tensor_parallel_size}")
        
        # 在有足够上下文长度的情况下，设置最大模型长度
        max_prompt_length = 16384
        max_response_length = data_config['max_response_length']
        max_model_len = max_prompt_length + max_response_length
        max_num_batched_tokens = max_model_len * 4  # 一个合理的批处理令牌数量
        
        print(f"Initializing vLLM model with tensor_parallel_size={tensor_parallel_size}")
        
        # 对于单GPU运行，设置tensor_parallel_size=1
        if tensor_parallel_size > 1 and world_size == 1:
            print("Only one GPU available, setting tensor_parallel_size=1")
            tensor_parallel_size = 1
        
        vllm_init_kwargs = {}
        limit_images = 10
        if limit_images > 0:
            vllm_init_kwargs = {"limit_mm_per_prompt": {"image": limit_images}}
        
        # 初始化LLM
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_sleep_mode=True,
            enable_chunked_prefill=enable_chunked_prefill,
            **vllm_init_kwargs
        )
        
        print("vLLM model initialized successfully")
        
        # 加载所有任务
        tasks = json.load(open('/home/zkliu/WR1/prompts/generate/expand_v2_shuffle.json', 'r'))
        
        for idx, task in enumerate(tasks):
            task['id'] = idx
        
        # 设置批处理大小
        batch_size = 16  # 可以根据系统资源调整
        
        # 构建meta_info
        meta_info = {
            "llm_type": "Qwen2_5_R1",
            "max_steps": 15,
            "temperature": 0.5,
            "n": 1,  # 每个任务生成的候选数量
            "max_tokens": 2048,  # 最大输出token数
            "history_n": 5
        }
        
        processor = AutoProcessor.from_pretrained(model_path)
        print("Processor loaded successfully")
        
        # 将任务分成批次
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        print(f"Total tasks: {len(tasks)}, will be processed in {total_batches} batches of size {batch_size}")
        
        # 处理每个批次
        for batch_idx in tqdm(range(total_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(tasks))
            current_batch = tasks[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_idx + 1}/{total_batches} (tasks {start_idx + 1}-{end_idx})")
            
            # 为当前批次创建代理状态
            agent_states = []
            init_tasks = []
            
            # 为当前批次的每个任务创建代理状态
            for task in current_batch:
                agent_state = AgentState(
                    task_id=task['id'],
                    task=task['query'],
                    agent_id=0,
                    start_url=task['start_url'],
                    level=task['level']
                )
                agent_state.output_dir = output_dir
                init_tasks.append((agent_state, None))
                agent_states.append(agent_state)
            
            print(f"Created {len(agent_states)} agent states for current batch")
            
            # 并行初始化当前批次的所有代理
            print("Initializing all agent instances for current batch...")
            
            async def init_agent_for_state(state, initial_response):
                try:
                    agent = ScreenshotAgentVLLM(
                        task=state.task,
                        llm_type=meta_info['llm_type'],
                        max_steps=meta_info['max_steps'],
                        start_url=f"https://{state.start_url}",  # 使用状态中保存的start_url
                        history_n=meta_info['history_n']
                    )
                    
                    initial_messages = await agent.setup()
                    
                    state.agent = agent
                    state.messages = initial_messages
                    state.steps = 0
                    
                    print(f"Agent {state.task_id} initialized")
                    return state
                except Exception as e:
                    print(f"Error initializing agent for state {state.task_id}_{state.agent_id}: {e}")
                    if state.agent:
                        await state.agent.cleanup()
                    state.completed = True
                    return state
            
            init_coros = [init_agent_for_state(state, response) for state, response in init_tasks]
            initialized_states = await asyncio.gather(*init_coros)
            
            # 开始多轮交互
            print(f"Starting multi-round interaction for current batch...")
            max_steps = meta_info['max_steps']
            
            for step in range(max_steps):
                print(f"\n----- Step {step+1}/{max_steps} for batch {batch_idx + 1} -----")
                
                active_states = [state for state in agent_states if not state.completed and state.agent is not None]
                
                if not active_states:
                    print("All agents in current batch completed their tasks or reached max steps")
                    break
                
                print(f"Current active agents in batch: {len(active_states)}")
                
                vllm_inputs = []
                
                for state in active_states:
                    prompt = processor.apply_chat_template(
                        state.messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    image_data, _ = process_vision_info(state.messages)
                    if image_data:
                        vllm_inputs.append({
                            'prompt': prompt,
                            'multi_modal_data': {'image': image_data}
                        })
                    else:
                        vllm_inputs.append({'prompt': prompt})
                
                if vllm_inputs:
                    outputs = llm.generate(vllm_inputs, sampling_params=SamplingParams(
                        temperature=meta_info['temperature'],
                        max_tokens=meta_info['max_tokens']
                    ))
                    
                    async def process_agent_response(state, response_text):
                        try:
                            next_messages, completed, answer = await state.agent.process_response(response_text)
                            state.messages = next_messages
                            state.completed = completed
                            state.answer = answer
                            
                            print(f"Agent {state.task_id}: {'Completed' if completed else 'Continuing'}")
                            if answer:
                                print(f"Agent {state.task_id} answer: {answer}")
                                
                        except Exception as e:
                            print(f"Error processing response for agent {state.unique_id}: {e}")
                            state.completed = True
                    
                    process_tasks = []
                    for i, output in enumerate(outputs):
                        state = active_states[i]
                        response_text = output.outputs[0].text
                        # print(response_text)
                        state.task_history.append(response_text)
                        state.steps += 1
                        process_tasks.append(process_agent_response(state, response_text))
                    
                    await asyncio.gather(*process_tasks)
                
                completed_count = sum(1 for state in agent_states if state.completed)
                print(f"Current batch progress: {completed_count}/{len(agent_states)} completed")
            
            # 清理当前批次的代理实例并保存历史记录
            print(f"Cleaning up and saving history for batch {batch_idx + 1}...")

            for state in agent_states:
                if state.agent:
                    await state.save_history()
                    await state.agent.cleanup()
            
            # if cleanup_tasks:
            #     await asyncio.gather(*cleanup_tasks)
            
            # 打印当前批次的结果
            print(f"\n----- Results for batch {batch_idx + 1} -----")
            for task in current_batch:
                task_states = [s for s in agent_states if s.task_id == task['id']]
                completed_states = [s for s in task_states if s.completed and s.answer]
                
                print(f"\nTask {task['id']}: {task['query']}")
                print(f"Total paths: {len(task_states)}, Completed with answers: {len(completed_states)}")
                
                if completed_states:
                    # print(f"Found answers for {task['id']}: {state.answer} (steps: {state.steps}) ")
                    for i, state in enumerate(completed_states):
                        print(f"  Task {state.task_id}: {state.answer} (steps: {state.steps})")
            
            # 在批次之间添加短暂延迟，让系统有机会释放资源
            if batch_idx < total_batches - 1:
                print("\nWaiting before processing next batch...")
                await asyncio.sleep(5)
                
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise



if __name__ == "__main__":
    # 设置多进程启动方法
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    # 创建事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # 运行测试
        loop.run_until_complete(test_generate_sequences_web())
        print("Program completed successfully.")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保清理所有资源
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()