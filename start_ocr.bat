@echo off
cd /d %~dp0
echo 开始图片 OCR 补录 (需 easyocr 模型, 首次联网下载; 失败会自动跳过, 可重跑)...
venv\Scripts\python.exe -u run_ocr.py
echo.
echo 图片补录完成。请重启界面 (start_app.bat) 或点侧边栏"重建 BM25 缓存" 使其生效。
pause
