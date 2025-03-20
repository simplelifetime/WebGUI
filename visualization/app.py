from flask import Flask, render_template, jsonify
import json
from pathlib import Path
import os

app = Flask(__name__)

def get_latest_session():
    """获取最新的会话日志"""
    logs_dir = Path("visualization/logs")
    if not logs_dir.exists():
        return None
    
    log_files = list(logs_dir.glob("session_*.json"))
    if not log_files:
        return None
    
    latest_log = max(log_files, key=os.path.getctime)
    with open(latest_log, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.route('/')
def index():
    """渲染主页"""
    session_data = get_latest_session()
    return render_template('index.html', session_data=session_data)

@app.route('/api/session')
def get_session():
    """获取会话数据的API端点"""
    session_data = get_latest_session()
    return jsonify(session_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 