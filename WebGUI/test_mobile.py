import asyncio
import os
from mobile_controller import MobileController

async def test_mobile_controller():
    # 创建测试目录
    test_dir = "test_outputs"
    os.makedirs(test_dir, exist_ok=True)
    
    # 初始化控制器
    controller = MobileController(
        device="emulator-5554",  # 替换为你的设备ID
        enable_logging=True,
        coordinate_system="absolute"
    )
    
    # 测试1: 获取设备尺寸
    print("\n测试1: 获取设备尺寸")
    width, height = controller.get_device_size()
    print(f"设备尺寸: {width}x{height}")
    
    # 测试2: 获取屏幕截图
    print("\n测试2: 获取屏幕截图")
    screenshot_path = await controller.get_screenshot("test_screenshot", test_dir)
    print(f"截图保存路径: {screenshot_path}")
    
    # 测试3: 点击操作
    print("\n测试3: 点击操作")
    tap_action = {
        "action_type": "tap",
        "action_inputs": {
            "start_box": [width//2, height//2]  # 点击屏幕中心
        }
    }
    result = await controller.execute_action(tap_action)
    print(f"点击结果: {result}")
    
    # 测试4: 文本输入
    print("\n测试4: 文本输入")
    type_action = {
        "action_type": "type",
        "action_inputs": {
            "content": "Hello World"
        }
    }
    result = await controller.execute_action(type_action)
    print(f"输入结果: {result}")
    
    # 测试5: 滑动操作
    print("\n测试5: 滑动操作")
    swipe_action = {
        "action_type": "swipe",
        "action_inputs": {
            "start_box": [width//2, height//2],
            "direction": "up"
        }
    }
    result = await controller.execute_action(swipe_action)
    print(f"滑动结果: {result}")
    
    # 测试6: 长按操作
    print("\n测试6: 长按操作")
    long_press_action = {
        "action_type": "long_press",
        "action_inputs": {
            "start_box": [width//2, height//2],
            "duration": 1000
        }
    }
    result = await controller.execute_action(long_press_action)
    print(f"长按结果: {result}")
    
    # 测试7: 返回操作
    print("\n测试7: 返回操作")
    back_action = {
        "action_type": "back",
        "action_inputs": {}
    }
    result = await controller.execute_action(back_action)
    print(f"返回结果: {result}")
    
    # 测试8: 等待操作
    print("\n测试8: 等待操作")
    wait_action = {
        "action_type": "wait",
        "action_inputs": {}
    }
    result = await controller.execute_action(wait_action)
    print(f"等待结果: {result}")
    
    # 测试9: 完成操作
    print("\n测试9: 完成操作")
    finish_action = {
        "action_type": "finished",
        "action_inputs": {}
    }
    result = await controller.execute_action(finish_action)
    print(f"完成结果: {result}")

if __name__ == "__main__":
    asyncio.run(test_mobile_controller()) 