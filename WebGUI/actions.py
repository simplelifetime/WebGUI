import re
from typing import Dict, Any, List, Optional, Tuple
from WebGUI.custom_types import FINISH_WORD, WAIT_WORD, ERROR_WORD

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

class R1ActionParser(ActionParser):
    """Parser for R1 prompt format responses"""
    
    @staticmethod
    def extract_tag_content(text: str, tag: str) -> Optional[str]:
        """Extract content between XML-like tags"""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def parse_actions(action_text: str) -> List[Dict[str, Any]]:
        """Parse multiple actions from action text"""
        actions = []
        action_lines = [line.strip() for line in action_text.split('\n') if line.strip()]
        
        for action_str in action_lines:
            parsed_action = R1ActionParser.parse_action(action_str)
            if parsed_action:
                actions.append(parsed_action)
                
        return actions

    @staticmethod
    def parse_llm_response(response: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Parse R1 format LLM response into structured actions and answer
        
        Returns:
            Tuple containing:
            - List of action dictionaries
            - Optional answer string (None if not present)
        """
        try:
            # Extract thought, action, and answer parts
            thought = R1ActionParser.extract_tag_content(response, "think")
            action_text = R1ActionParser.extract_tag_content(response, "action")
            answer = R1ActionParser.extract_tag_content(response, "answer")
            
            if not thought or not action_text:
                # print("Missing required thought or action sections")
                return [{"action_type": ERROR_WORD}], None
                
            # Parse actions
            parsed_actions = R1ActionParser.parse_actions(action_text)
            if not parsed_actions:
                return [{"action_type": ERROR_WORD}], None
                
            # Format actions with thought
            formatted_actions = []
            for action in parsed_actions:
                action_dict = {
                    "thought": thought,
                    "action_type": action["function"],
                    "action_inputs": action["args"]
                }
                formatted_actions.append(action_dict)
            
            return formatted_actions, answer
            
        except Exception as e:
            print(f"Error parsing R1 LLM response: {e}")
            print(f"Raw response: {response}")
            return [{"action_type": ERROR_WORD}], None 