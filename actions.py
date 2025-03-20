import re
from typing import Dict, Any, List
from custom_types import FINISH_WORD, WAIT_WORD, ERROR_WORD

class ActionParser:
    @staticmethod
    def parse_action(action_str: str) -> Dict[str, Any]:
        """Parse a single action string into structured format"""
        try:
            action_str = action_str.strip()
            
            # Handle simple actions
            if action_str in [WAIT_WORD, FINISH_WORD]:
                return {
                    'function': action_str,
                    'args': {}
                }
                
            # Parse function call format: action(param='value')
            match = re.match(r"(\w+)\((.*)\)", action_str)
            if not match:
                print(f"Failed to parse action: {action_str}")
                return None
                
            func_name = match.group(1)
            params_str = match.group(2)
            
            # Parse parameters
            params = {}
            # Split by comma but not within quotes
            param_pairs = re.findall(r"(\w+)='([^']*)'", params_str)
            
            for key, value in param_pairs:
                if key in ["start_box", "end_box"]:
                    # Convert (x,y) format to [x,y] format
                    coords = value.strip("()").split(",")
                    if len(coords) == 2:
                        x = float(coords[0].strip())
                        y = float(coords[1].strip())
                        value = [x, y]
                params[key] = value
                
            return {
                'function': func_name,
                'args': params
            }
            
        except Exception as e:
            print(f"Failed to parse action '{action_str}': {e}")
            return None

    @staticmethod
    def parse_llm_response(response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured actions"""
        try:
            # Extract thought and action parts
            thought_match = re.search(r"Thought: (.+?)(?=\s*Action:|$)", response, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""
            
            actions = []
            if "Action:" in response:
                # Get all text after "Action:"
                action_text = response.split("Action:")[-1].strip()
                
                # Split by newlines and filter out empty lines
                action_lines = [line.strip() for line in action_text.split('\n') if line.strip()]
                
                # Parse each action line
                for action_str in action_lines:
                    parsed_action = ActionParser.parse_action(action_str)
                    if parsed_action:
                        action_dict = {
                            "thought": thought,
                            "action_type": parsed_action["function"],
                            "action_inputs": parsed_action["args"]
                        }
                        actions.append(action_dict)
            
            return actions if actions else [{"action_type": ERROR_WORD}]
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response}")
            return [{"action_type": ERROR_WORD}] 