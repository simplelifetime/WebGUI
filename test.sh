
# 使用UITARS模型（默认）
# CUDA_VISIBLE_DEVICES=8,9 python main.py --task "请帮我查找明天东方航空从北京到上海的航班" --max_steps 15 --enable_logging

CUDA_VISIBLE_DEVICES=8,9 python main.py --task "请帮我搜索姚明的图片" --max_steps 10 --enable_logging --llm Qwen2_5

# 使用gemma模型
# python main.py --task "Search all the teachers in 中国人民大学 高瓴人工智能学院." --llm gemma --max_steps 10

# 指定最大步数和起始URL
# python main.py --task "Find information about AI research" --llm gemma --max_steps 15 --start_url "https://www.google.com"

# start docker
# docker run -p 3000:3000 --rm --init -it --workdir /home/pwuser --user pwuser mcr.microsoft.com/playwright:v1.51.0-noble /bin/sh -c "npx -y playwright@1.51.0 run-server --port 3000 --host 0.0.0.0 --headed"