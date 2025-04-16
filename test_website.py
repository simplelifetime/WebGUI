from WebGUI.screenshot_agent_vllm import ScreenshotAgentVLLM
import os
os.environ['https_proxy'] = "http://127.0.0.1:7890"
os.environ['http_proxy'] = "http://127.0.0.1:7890"
from PIL import Image
from io import BytesIO
import base64
import json
import asyncio
websites = json.load(open('website_dict_v2.json'))


meta_info = {
    "llm_type": "Qwen2_5_R1",
    "max_steps": 8,
    "start_url": "https://www.google.com/",
    "temperature": 0.5,
    "n": 1,  # 每个任务生成的候选数量
    "max_tokens": 2048,  # 最大输出token数
    "history_n" : 8
}

async def main():
    from openai import OpenAI
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    
    agent = ScreenshotAgentVLLM(
        task="What's this website about?",
        llm_type=meta_info['llm_type'],
        max_steps=meta_info['max_steps'],
        start_url='https://www.arxiv.org/',
        history_n=meta_info['history_n']
    )

    await agent.setup()  # Initialize browser and page
    new_website_dict = {}
    cnt = 0
    for domain in websites:
        new_website_dict[domain] = []
        for web in websites[domain]:
            passr = 1
            try:
                if web.startswith('http'):
                    weburl = web
                else:
                    weburl = f'https://{web}'
                response = await agent.browser_controller.page.goto(weburl, timeout=5000)
                screenshot, _ = await agent.browser_controller.get_screenshots()
                chat_response = client.chat.completions.create(
                    model="/mnt/data/zkliu/hf_models/qwen2_5-vl-32b-instruct",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{screenshot}"
                                    },
                                },
                                {"type": "text", "text": "Please determine if the website requires login to proceed most of its functions. If so, print yes, otherwise print no. Do not print anything else. If the page is blank or require captcha, also print yes."},
                            ],
                        },
                    ],
                )
                
                print(f"\n{chat_response.choices[0].message.content}\n")
                
                if chat_response.choices[0].message.content.lower() == "yes":
                    passr = 0
                    
            except Exception as e:
                # print(f"Error in action execution: {e}")
                passr = 0
            if passr:
                new_website_dict[domain].append(web)
                print(f"{web} is available")
            else:
                print(f"{web} is not available")
                
            cnt += passr
            
    print(f"Total available websites: {cnt}")
    json.dump(new_website_dict, open('website_dict_v2_filtered.json', 'w'), indent=4)
    return new_website_dict

if __name__ == "__main__":
    new_website_dict = asyncio.run(main())
