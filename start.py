"""
启动助手: 读取 config.yaml 的 server 配置, 拉起 Streamlit 局域网界面。
用法:
  python start.py
(也可直接: venv/Scripts/python.exe -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501)
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SERVER


def main():
    host = SERVER.get("host", "0.0.0.0")
    port = int(SERVER.get("port", 8501))
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py",
           "--server.address", str(host), "--server.port", str(port)]
    print("启动 bjhc 知识库界面 ->", " ".join(cmd))
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
