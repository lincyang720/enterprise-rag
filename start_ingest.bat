@echo off
cd /d %~dp0
echo 开始入库 (文本类, 跳过图片OCR; 支持断点续传, 中断后重跑会自动跳过已完成文件)...
venv\Scripts\python.exe -u run_ingest.py --no-images
echo.
echo 入库完成。如需把图片也 OCR 进库, 另开一个窗口运行 start_ocr.bat
pause
