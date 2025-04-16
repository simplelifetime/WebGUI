"""
Configuration settings for the prompt generation system.
"""

# API Configuration
# OPENAI_API_KEY = "sk-759a521b142648fd9c37c1b00974a308"
# OPENAI_BASE_URL = "https://api.deepseek.com"
# MODEL_NAME = "deepseek-chat"

OPENAI_API_KEY = 'sk-eade454dbee344d8a8d8658359e7b9c8'
OPENAI_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL_NAME = "qwen-plus"


# File paths
WEBSITES_JSON_PATH = "../website_dict_v2_filtered.json"
OUTPUT_DIR = "generate"
OUTPUT_FILE = "expand_v2.json"
NUM_QUERIES = 1000
SEED_PATH = "generate/seed.json"