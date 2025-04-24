import os
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
import base64
import asyncio
from .custom_types import FINISH_WORD, WAIT_WORD, ERROR_WORD
from .actions import ActionParser, R1ActionParser

class MobileController:
    def __init__(self, device: str, enable_logging: bool = False, coordinate_system: str = "absolute"):
        self.device = device
        self.enable_logging = enable_logging
        self.coordinate_system = coordinate_system
        self.screenshot_dir = "/sdcard/AppAgent/screenshots"
        self.width, self.height = self.get_device_size()
        self.backslash = "\\"
        self.cursor_x = 0
        self.cursor_y = 0

    def execute_adb(self, adb_command: str) -> str:
        """Execute ADB command and return result"""
        result = subprocess.run(adb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        print(f"Command execution failed: {adb_command}")
        print(result.stderr)
        return "ERROR"

    def get_device_size(self) -> Tuple[int, int]:
        """Get device screen size"""
        adb_command = f"adb -s {self.device} shell wm size"
        result = self.execute_adb(adb_command)
        if result != "ERROR":
            return map(int, result.split(": ")[1].split("x"))
        return 0, 0

    def _convert_coordinates(self, x: float, y: float) -> Tuple[int, int]:
        """Convert coordinates based on coordinate system"""
        if self.coordinate_system == "relative":
            return int(x * self.width), int(y * self.height)
        return int(x), int(y)

    async def get_screenshot(self, prefix: str, save_dir: str) -> str:
        """Get screenshot from device"""
        cap_command = f"adb -s {self.device} shell screencap -p " \
                     f"{os.path.join(self.screenshot_dir, prefix + '.png').replace(self.backslash, '/')}"
        pull_command = f"adb -s {self.device} pull " \
                      f"{os.path.join(self.screenshot_dir, prefix + '.png').replace(self.backslash, '/')} " \
                      f"{os.path.join(save_dir, prefix + '.png')}"
        result = self.execute_adb(cap_command)
        if result != "ERROR":
            result = self.execute_adb(pull_command)
            if result != "ERROR":
                return os.path.join(save_dir, prefix + ".png")
        return result

    async def execute_action(self, action: Dict[str, Any]) -> str:
        """Execute mobile action"""
        action_type = action.get("action_type", "").lower()
        action_inputs = action.get("action_inputs", {})
        
        try:
            if self.enable_logging:
                print(f"Executing action: {action_type} with inputs: {action_inputs}")
            
            if action_type == "tap":
                if "start_box" in action_inputs:
                    box = action_inputs["start_box"]
                    x, y = self._convert_coordinates(box[0], box[1])
                    self.cursor_x = x
                    self.cursor_y = y
                    adb_command = f"adb -s {self.device} shell input tap {x} {y}"
                    return self.execute_adb(adb_command)
                    
            elif action_type == "type":
                if "content" in action_inputs:
                    content = action_inputs["content"]
                    content = content.replace(" ", "%s").replace("'", "")
                    adb_command = f"adb -s {self.device} shell input text {content}"
                    return self.execute_adb(adb_command)
                    
            elif action_type == "swipe":
                if "start_box" in action_inputs and "direction" in action_inputs:
                    box = action_inputs["start_box"]
                    direction = action_inputs["direction"]
                    x, y = self._convert_coordinates(box[0], box[1])
                    
                    unit_dist = int(self.width / 10)
                    if direction == "up":
                        offset = 0, -2 * unit_dist
                    elif direction == "down":
                        offset = 0, 2 * unit_dist
                    elif direction == "left":
                        offset = -1 * unit_dist, 0
                    elif direction == "right":
                        offset = unit_dist, 0
                    else:
                        return ERROR_WORD
                        
                    duration = 400
                    adb_command = f"adb -s {self.device} shell input swipe {x} {y} {x+offset[0]} {y+offset[1]} {duration}"
                    return self.execute_adb(adb_command)
                    
            elif action_type == "long_press":
                if "start_box" in action_inputs:
                    box = action_inputs["start_box"]
                    x, y = self._convert_coordinates(box[0], box[1])
                    duration = action_inputs.get("duration", 1000)
                    adb_command = f"adb -s {self.device} shell input swipe {x} {y} {x} {y} {duration}"
                    return self.execute_adb(adb_command)
                    
            elif action_type == "back":
                adb_command = f"adb -s {self.device} shell input keyevent KEYCODE_BACK"
                return self.execute_adb(adb_command)
                
            elif action_type == "wait":
                await asyncio.sleep(2)
                return WAIT_WORD
                
            elif action_type == "finished":
                return FINISH_WORD
                
            return ERROR_WORD
            
        except Exception as e:
            print(f"Error executing action: {e}")
            return ERROR_WORD 