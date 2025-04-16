"""
Prompt generation module for simulating human operations on GUI interfaces.
This module generates queries that would be performed on target webpages.
"""

from typing import Dict, List, Optional, Tuple
import json
import random
import os
import time
from tqdm import tqdm
from openai import OpenAI
import multiprocessing
from config import (
    OPENAI_API_KEY, 
    OPENAI_BASE_URL, 
    MODEL_NAME, 
    WEBSITES_JSON_PATH,
    OUTPUT_DIR,
    NUM_QUERIES,
    OUTPUT_FILE,
    SEED_PATH
)

def process_instance(args: Tuple[str, str, str, int]) -> Optional[Dict]:
    """Process a single instance to generate a query.
    
    Args:
        args: Tuple containing (prompt, domain, start_url)
        
    Returns:
        Optional[Dict]: Generated query data or None if failed
    """
    prompt, domain, start_url, query_id, level = args
    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        
        # Retry mechanism
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    temperature=1.5,
                    max_tokens=150,
                )
                return {
                    "id": query_id,
                    "domain": domain,
                    "start_url": start_url,
                    "query": response.choices[0].message.content.strip(),
                    "level": level
                }
            except Exception as e:
                if attempt == 2:  # Last attempt
                    print(f"Error generating query {query_id}: {str(e)}")
                    return None
                time.sleep(1 * (attempt + 1))
    except Exception as e:
        print(f"Error in process_instance: {str(e)}")
        return None

class PromptGenerator:
    """Class for generating prompts for simulated GUI operations."""
    
    def __init__(self, prompt_template_path: str = "prompt_templates.json"):
        """Initialize the PromptGenerator with website data and prompt templates.
        
        Args:
            prompt_template_path (str): Path to the prompt templates JSON file
        """
        self.websites = self._load_websites()
        self.prompt_templates = self._load_prompt_templates(prompt_template_path)
        self.cur_web_list = json.load(open(SEED_PATH, 'r'))

    def _load_prompt_templates(self, template_path: str) -> Dict:
        """Load prompt templates from JSON file.
        
        Args:
            template_path (str): Path to the prompt templates JSON file
            
        Returns:
            Dict: Loaded prompt templates
        """
        try:
            with open(template_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt templates file not found at {template_path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError("Invalid JSON format in prompt templates file")

    def _load_websites(self) -> Dict:
        """Load website data from JSON file."""
        try:
            with open(WEBSITES_JSON_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Websites JSON file not found at {WEBSITES_JSON_PATH}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError("Invalid JSON format in websites file")

    def get_random_website(self) -> tuple[str, str]:
        """Get a random domain and start URL from the websites data."""
        domain = random.sample(list(self.websites.keys()), 1)[0]
        start_url = random.sample(self.websites[domain], 1)[0]
        return domain, start_url

    def generate_prompt(self, type='width') -> Tuple[str, str, str]:
        """Generate a complete prompt with random website data."""
        data_sample = random.sample(self.cur_web_list, 1)[0]
        domain = data_sample['domain']
        start_url = data_sample['start_url']
        query = data_sample['query']
        level = data_sample['level']
        if type=='width':
            prompt = self.prompt_templates["width_prompt"]["user_prompt_template"].format(
                start_url=start_url, 
                domain=domain,
                query=query
            )
            level = level
        elif type=='depth':
            prompt = self.prompt_templates["depth_prompt"]["user_prompt_template"].format(
                start_url=start_url, 
                domain=domain,
                query=query
            )
            level = level + 1
        return prompt, domain, start_url, level
    
    def generate_multiround_query(self, num_rounds: int, queries_per_round: int) -> List[Dict]:
        """Generate multiple rounds of queries with width and depth variations.
        
        Args:
            num_rounds (int): Number of rounds to generate queries
            queries_per_round (int): Number of queries to generate per round
            
        Returns:
            List[Dict]: List of all generated queries
        """
        all_queries = []
        
        for round_num in range(num_rounds):
            print(f"Starting round {round_num + 1}/{num_rounds}")
            
            # Generate prompts for this round
            prompts = []
            for _ in range(min(queries_per_round, len(self.cur_web_list))):
                # Randomly select a data sample
                # 80% width, 20% depth
                query_type = 'width' if random.random() < 0.6 else 'depth'
                
                # Generate prompt
                prompt, domain, start_url, level = self.generate_prompt(type=query_type)
                prompts.append((prompt, domain, start_url, len(all_queries) + len(prompts), level))
            
            # Generate queries in parallel
            
            num_processes = min(16, multiprocessing.cpu_count() * 2)  # Use up to 32 processes
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(process_instance, prompts),
                    total=len(prompts),
                    desc="Generating queries"
                ))
            
            round_queries = [r for r in results if r is not None]
            
            # Add new queries to cur_web_list and all_queries
            self.cur_web_list.extend(round_queries)
            all_queries.extend(round_queries)
            
            print(f"Round {round_num + 1} completed. Generated {len(round_queries)} queries.")
        
        return all_queries
    
    def generate_seed_queries(self) -> List[Dict]:
        prompts = []
        level = 1
        for domain in self.websites.keys():
            for start_url in self.websites[domain]:
                for i in range(3):
                    prompts.append((self.prompt_templates["base_prompt"]["user_prompt_template"].format(
                        start_url=start_url, 
                        domain=domain
                    ), domain, start_url, 0, level))

        num_processes = min(16, multiprocessing.cpu_count() * 2)
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_instance, prompts),
                total=len(prompts),
                desc="Generating queries"
            ))
        
        # Filter out None results and return valid queries
        return [r for r in results if r is not None]
        

    def save_queries(self, queries: List[Dict], output_dir: str = OUTPUT_DIR, output_file: str = OUTPUT_FILE):
        """Save generated queries to JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(queries)} queries to {output_file}")

def main():
    """Main function to generate and save multiple queries."""
    try:
        generator = PromptGenerator()
        print(f"Generating {NUM_QUERIES} queries in parallel...")
        # queries = generator.generate_seed_queries()
        queries = generator.generate_multiround_query(num_rounds=4, queries_per_round=800)
        generator.save_queries(queries)
        print("Query generation completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()