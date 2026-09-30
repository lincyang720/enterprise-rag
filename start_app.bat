@echo off
cd /d %~dp0
echo 正在启动 bjhc 知识库界面 (局域网可访问)...
venv\Scripts\python.exe -u start.py
pause
