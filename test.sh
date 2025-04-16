
# 使用UITARS模型（默认）
# CUDA_VISIBLE_DEVICES=8,9 python main.py --task "请帮我查找明天东方航空从北京到上海的航班" --max_steps 15 --enable_logging

# python WebGUI/main.py --task "2023年F1赛事的年度总冠军是谁？" --max_steps 10 --enable_logging --llm Qwen2_5_R1 --max_steps 10

# python main.py --task "搜索当下最近的F1赛事，并总结一下赛事结果" --max_steps 10 --enable_logging --llm Qwen2_5

# python main.py --task "请帮我找到一张姚明穿着11号队服的图片" --max_steps 10 --enable_logging --llm Qwen2_5

# 使用gemma模型
# python main.py --task "Search all the teachers in 中国人民大学 高瓴人工智能学院." --llm gemma --max_steps 10

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# torchrun --nproc_per_node=1 --master_port=29500 test.py
python test.py