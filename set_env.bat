@echo off
chcp 65001 >nul
:: ============================================================
:: 将 HuggingFace 模型缓存目录设置到 D 盘，并配置国内镜像（hf-mirror.com）
:: 双击此文件即可完成设置（用 setx 持久化到用户环境变量）
:: ============================================================

echo.
echo 正在设置 HuggingFace 模型缓存到 D 盘...

:: 1. 创建缓存目录
if not exist "D:\huggingface\cache"                 mkdir "D:\huggingface\cache"
if not exist "D:\huggingface\transformers"          mkdir "D:\huggingface\transformers"
if not exist "D:\huggingface\sentence-transformers" mkdir "D:\huggingface\sentence-transformers"

:: 2. 持久化环境变量（写入当前用户，重新打开终端后生效）
setx HF_HOME "D:\huggingface\cache" >nul
setx TRANSFORMERS_CACHE "D:\huggingface\transformers" >nul
setx SENTENCE_TRANSFORMERS_HOME "D:\huggingface\sentence-transformers" >nul
setx HF_ENDPOINT "https://hf-mirror.com" >nul

:: 3. 同时写入当前会话（本窗口内立即生效）
set "HF_HOME=D:\huggingface\cache"
set "TRANSFORMERS_CACHE=D:\huggingface\transformers"
set "SENTENCE_TRANSFORMERS_HOME=D:\huggingface\sentence-transformers"
set "HF_ENDPOINT=https://hf-mirror.com"

echo.
echo [完成] 已设置以下用户环境变量:
echo   HF_HOME                     = D:\huggingface\cache
echo   TRANSFORMERS_CACHE          = D:\huggingface\transformers
echo   SENTENCE_TRANSFORMERS_HOME  = D:\huggingface\sentence-transformers
echo   HF_ENDPOINT                 = https://hf-mirror.com
echo.
echo 提示: 环境变量已持久化，请在"新的"终端或 IDE 里运行 python 才会生效。
echo 模型将通过国内镜像下载，已缓存的模型直接本地加载。
echo.
pause
