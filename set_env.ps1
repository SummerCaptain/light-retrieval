# ============================================================
# 将 HuggingFace 模型缓存目录改为 D 盘
#
# 使用方式:
#   方式1 (推荐): 双击 set_env.bat 自动绕过执行策略
#   方式2 (手动): 在 PowerShell 中执行:
#       powershell -ExecutionPolicy Bypass -File .\set_env.ps1
# ============================================================

$CacheDir = "D:\huggingface\cache"
$TransformersDir = "D:\huggingface\transformers"
$STDir = "D:\huggingface\sentence-transformers"

# 创建目录
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $TransformersDir | Out-Null
New-Item -ItemType Directory -Force -Path $STDir | Out-Null

# 设置当前会话环境变量
$env:HF_HOME = $CacheDir
$env:TRANSFORMERS_CACHE = $TransformersDir
$env:SENTENCE_TRANSFORMERS_HOME = $STDir
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 持久化到系统环境变量 (当前用户)
[Environment]::SetEnvironmentVariable("HF_HOME", $CacheDir, "User")
[Environment]::SetEnvironmentVariable("TRANSFORMERS_CACHE", $TransformersDir, "User")
[Environment]::SetEnvironmentVariable("SENTENCE_TRANSFORMERS_HOME", $STDir, "User")
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")

Write-Host ""
Write-Host "HuggingFace 缓存目录已设置为 D 盘，并配置国内镜像" -ForegroundColor Green
Write-Host "  HF_HOME                     = $CacheDir"
Write-Host "  TRANSFORMERS_CACHE          = $TransformersDir"
Write-Host "  SENTENCE_TRANSFORMERS_HOME  = $STDir"
Write-Host "  HF_ENDPOINT                 = https://hf-mirror.com"
Write-Host ""
Write-Host "模型将通过国内镜像下载（hf-mirror.com），已缓存的模型直接本地加载。" -ForegroundColor Cyan
Write-Host ""