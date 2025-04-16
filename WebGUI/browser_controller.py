import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from WebGUI.custom_types import FINISH_WORD, WAIT_WORD, ERROR_WORD
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from PIL import Image

class BrowserController:
    def __init__(self, enable_logging=False, coordinate_system="relative"):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page = None
        self.viewport = {'width': 1280, 'height': 720}
        self.cursor_x = 0
        self.cursor_y = 0
        
        # 坐标系统配置：'relative' 表示 0-1000 的相对坐标，'absolute' 表示绝对坐标
        self.coordinate_system = coordinate_system
        
        # 是否启用日志记录
        self.enable_logging = enable_logging
        
        if self.enable_logging:
            # 创建可视化相关的目录
            self.visualization_dir = Path("visualization")
            self.visualization_dir.mkdir(exist_ok=True)
            self.screenshots_dir = self.visualization_dir / "screenshots"
            self.screenshots_dir.mkdir(exist_ok=True)
            self.logs_dir = self.visualization_dir / "logs"
            self.logs_dir.mkdir(exist_ok=True)
            
            # 初始化日志文件
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.logs_dir / f"session_{self.session_id}.json"
            self.interaction_log = []

    async def setup(self):
        """Setup browser and page"""
        # 创建浏览器配置
        browser_config = BrowserConfig(
            headless=True,  # 设置为False以显示浏览器窗口
            disable_security=True,
        )
        
        # 创建上下文配置
        context_config = BrowserContextConfig(
            browser_window_size=self.viewport,
            wait_between_actions=5.0,
            minimum_wait_page_load_time=5.0,
            wait_for_network_idle_page_load_time=5.0,
            maximum_wait_page_load_time=10.0
        )
        
        # 初始化浏览器和上下文
        self.browser = Browser(config=browser_config)
        self.context = BrowserContext(browser=self.browser, config=context_config)
        await self.context._initialize_session()
        self.page = await self.context.get_current_page()
        
        # 记录初始状态
        if self.enable_logging:
            await self._log_interaction("setup", "Browser initialized")

    async def _log_interaction(self, action_type: str, details: Dict[str, Any]):
        """记录交互日志"""
        if not self.enable_logging:
            return
            
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "action_type": action_type,
            "details": details,
            "cursor_position": {"x": self.cursor_x, "y": self.cursor_y}
        }
        
        # 保存截图
        # if self.page:
        #     screenshot_path = self.screenshots_dir / f"{timestamp.replace(':', '_')}.png"
        #     await self.page.screenshot(path=str(screenshot_path))
        #     log_entry["screenshot"] = str(screenshot_path)
            
        #     # 创建GIF
        #     try:
        #         images = []
        #         for img_path in sorted(self.screenshots_dir.glob("*.png")):
        #             img = Image.open(img_path)
        #             images.append(img)
                
        #         if images:
        #             gif_path = self.visualization_dir / f"session_{self.session_id}.gif"
        #             images[0].save(
        #                 gif_path,
        #                 save_all=True,
        #                 append_images=images[1:],
        #                 duration=500,  # 每帧显示时间，单位为毫秒
        #                 loop=0  # 无限循环
        #             )
        #             print(f"GIF updated at {gif_path}")
        #     except Exception as e:
        #         print(f"Error creating GIF: {e}")
        
        self.interaction_log.append(log_entry)
        
        # 实时写入日志文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.interaction_log, f, indent=2, ensure_ascii=False)

    async def get_current_page(self):
        """获取当前活动页面"""
        if not self.context:
            raise RuntimeError("Browser context not initialized")
        
        # 获取当前活动页面
        current_page = await self.context.get_current_page()
        if current_page != self.page:
            # print(f"Page changed from {self.page.url if self.page else 'None'} to {current_page.url}")
            self.page = current_page
        return self.page

    async def get_screenshots(self) -> Tuple[str, str]:
        """Take both normal and cursor-highlighted screenshots"""
        page = await self.get_current_page()
        if not page:
            raise RuntimeError("Browser page not initialized")
            
        # 获取普通截图
        try:
            normal_screenshot = await self.context.take_screenshot()
        except:
            print("Warning: Page load timed out, proceeding with screenshot.")
            normal_screenshot = await page.screenshot(
                full_page=False,
                animations='disabled',
                timeout=5000
            )
            normal_screenshot = base64.b64encode(normal_screenshot).decode('utf-8')
        
        # 获取带光标的截图
        await page.evaluate(f"""
            const div = document.createElement('div');
            div.style.position = 'absolute';
            div.style.left = '{self.cursor_x - 10}px';
            div.style.top = '{self.cursor_y - 10}px';
            div.style.width = '20px';
            div.style.height = '20px';
            div.style.backgroundColor = 'rgba(255, 0, 0, 0.5)';
            div.style.zIndex = '9999';
            document.body.appendChild(div);
        """)
        
        try:
            cursor_screenshot = await self.context.take_screenshot()
        except:
            cursor_screenshot = await page.screenshot(
                full_page=False,
                animations='disabled',
            )
            cursor_screenshot = base64.b64encode(cursor_screenshot).decode('utf-8')
        
        await page.evaluate("""
            const div = document.querySelector('div[style*="z-index: 9999"]');
            if (div) div.remove();
        """)
        
        # 记录截图操作
        if self.enable_logging:
            await self._log_interaction("screenshot", {
                "type": "both",
                "cursor_position": {"x": self.cursor_x, "y": self.cursor_y}
            })
        
        return normal_screenshot, cursor_screenshot

    def _convert_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """
        统一转换坐标系统
        :param x: 输入的x坐标
        :param y: 输入的y坐标
        :return: 转换后的(x, y)坐标元组
        """
        if self.coordinate_system == "relative":
            # 将0-1000的相对坐标转换为绝对坐标
            converted_x = round(float(x) / 1000 * self.viewport["width"], 3)
            converted_y = round(float(y) / 1000 * self.viewport["height"], 3)
        else:
            # 对于绝对坐标，直接使用输入值
            converted_x = round(float(x), 3)
            converted_y = round(float(y), 3)
        return converted_x, converted_y

    async def execute_action(self, action: Dict[str, Any]) -> str:
        """Execute browser action"""
        page = await self.get_current_page()
        if not page:
            raise RuntimeError("Browser page not initialized")
            
        action_type = action.get("action_type", "").lower()
        action_inputs = action.get("action_inputs", {})
        
        try:
            # 记录动作开始
            if self.enable_logging:
                await self._log_interaction("action_start", {
                    "action_type": action_type,
                    "action_inputs": action_inputs
                })
            
            if action_type == "click":
                if "start_box" in action_inputs:
                    box = action_inputs["start_box"]
                    x, y = self._convert_coordinates(box[0], box[1])
                    self.cursor_x = x
                    self.cursor_y = y
                    await page.mouse.click(x, y)
                else:
                    await page.mouse.click()
                    
            elif action_type == "type":
                if "content" in action_inputs:
                    content = action_inputs["content"]
                    await page.keyboard.type(content.strip('\\n'))
                    if content.endswith("\\n"):
                        await page.keyboard.press("Enter")
                    
            elif action_type == "hotkey":
                if "key" in action_inputs:
                    keys = action_inputs["key"].split()
                    for key in keys:
                        await page.keyboard.down(key.capitalize())
                    for key in reversed(keys):
                        await page.keyboard.up(key.capitalize())
                    
            elif action_type == "left_double":
                if "start_box" in action_inputs:
                    box = action_inputs["start_box"]
                    x, y = self._convert_coordinates(box[0], box[1])
                    self.cursor_x = x
                    self.cursor_y = y
                    await page.mouse.dblclick(x, y)
                    
            elif action_type == "right_single":
                if "start_box" in action_inputs:
                    box = action_inputs["start_box"]
                    x, y = self._convert_coordinates(box[0], box[1])
                    self.cursor_x = x
                    self.cursor_y = y
                    await page.mouse.click(x, y, button="right")
                    
            elif action_type == "drag":
                if "start_box" in action_inputs and "end_box" in action_inputs:
                    start_box = action_inputs["start_box"]
                    end_box = action_inputs["end_box"]
                    start_x, start_y = self._convert_coordinates(start_box[0], start_box[1])
                    end_x, end_y = self._convert_coordinates(end_box[0], end_box[1])
                    self.cursor_x = start_x
                    self.cursor_y = start_y
                    await page.mouse.move(start_x, start_y)
                    await page.mouse.down()
                    self.cursor_x = end_x
                    self.cursor_y = end_y
                    await page.mouse.move(end_x, end_y)
                    await page.mouse.up()
                    
            elif action_type == "scroll":
                if "start_box" in action_inputs and "direction" in action_inputs:
                    box = action_inputs["start_box"]
                    x, y = self._convert_coordinates(box[0], box[1])
                    self.cursor_x = x
                    self.cursor_y = y
                    direction = action_inputs["direction"].lower()
                    if "up" in direction:
                        await page.mouse.wheel(0, -720)
                    elif "down" in direction:
                        await page.mouse.wheel(0, 720)
                        
            elif action_type == "goto":
                await page.goto(action_inputs["url"], timeout=10000)
                
            elif action_type == 'goback':
                await page.go_back()
                        
            elif action_type in [FINISH_WORD, WAIT_WORD, ERROR_WORD]:
                return action_type
                
            # 记录动作完成
            if self.enable_logging:
                await self._log_interaction("action_complete", {
                    "action_type": action_type,
                    "status": "success"
                })
                
        except Exception as e:
            print(f"Error executing action: {e}")
            # 记录动作失败
            if self.enable_logging:
                await self._log_interaction("action_error", {
                    "action_type": action_type,
                    "error": str(e)
                })
            return ERROR_WORD
            
        return "success"

    async def close(self):
        """Close the browser"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
        except Exception as e:
            print(f"Warning: Error during browser cleanup: {e}")
        finally:
            self.context = None
            self.browser = None
            self.page = None
            
            # 记录会话结束
            if self.enable_logging:
                await self._log_interaction("session_end", {
                    "status": "completed",
                    "total_actions": len(self.interaction_log)
                })
            
    def set_logging(self, enable: bool):
        """设置是否启用日志记录"""
        if enable and not self.enable_logging:
            # 如果之前未启用日志，现在要启用，需要初始化相关目录和文件
            self.visualization_dir = Path("visualization")
            self.visualization_dir.mkdir(exist_ok=True)
            self.screenshots_dir = self.visualization_dir / "screenshots"
            self.screenshots_dir.mkdir(exist_ok=True)
            self.logs_dir = self.visualization_dir / "logs"
            self.logs_dir.mkdir(exist_ok=True)
            
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.logs_dir / f"session_{self.session_id}.json"
            self.interaction_log = []
            
        self.enable_logging = enable 