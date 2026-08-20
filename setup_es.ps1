# ============================================================
# Elasticsearch 本地环境配置脚本 (Windows)
# ============================================================
# 用法: powershell -ExecutionPolicy Bypass -File setup_es.ps1
# 说明: 自动下载 ES 8.x 到本地 data 目录，无需安装 Java
#       ES 8.x 自带 JDK 17，解压即可运行
# ============================================================

$ES_VERSION = "8.15.0"
$ES_DIR = Join-Path $PSScriptRoot "data\elasticsearch-$ES_VERSION"
$ES_DOWNLOAD_URL = "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-$ES_VERSION-windows-x86_64.zip"
$ES_ZIP = Join-Path $PSScriptRoot "data\elasticsearch.zip"

Write-Host "=== Elasticsearch 本地环境配置 ===" -ForegroundColor Cyan
Write-Host ""

# ----------------------------------------------------------
# 1. 检查是否已安装
# ----------------------------------------------------------
if (Test-Path (Join-Path $ES_DIR "bin\elasticsearch.bat")) {
    Write-Host "[已存在] ES $ES_VERSION 已解压到 $ES_DIR" -ForegroundColor Green
}
else {
    Write-Host "[下载] 正在下载 ES $ES_VERSION ..." -ForegroundColor Yellow
    Write-Host "  URL: $ES_DOWNLOAD_URL" -ForegroundColor Gray

    # 确保 data 目录存在
    $null = New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "data")

    try {
        # 下载 zip
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $ES_DOWNLOAD_URL -OutFile $ES_ZIP -ErrorAction Stop
        Write-Host "[完成] 下载完成" -ForegroundColor Green

        # 解压
        Write-Host "[解压] 正在解压..." -ForegroundColor Yellow
        Expand-Archive -Path $ES_ZIP -DestinationPath (Join-Path $PSScriptRoot "data") -Force
        Remove-Item $ES_ZIP -Force
        Write-Host "[完成] 解压完成" -ForegroundColor Green
    }
    catch {
        Write-Host "[错误] 下载失败: $_" -ForegroundColor Red
        Write-Host ""
        Write-Host "请手动下载 ES:" -ForegroundColor Yellow
        Write-Host "  1. 访问: https://www.elastic.co/downloads/elasticsearch" -ForegroundColor White
        Write-Host "  2. 下载 Windows 版本 (elasticsearch-$ES_VERSION-windows-x86_64.zip)" -ForegroundColor White
        Write-Host "  3. 解压到 data\ 目录" -ForegroundColor White
        exit 1
    }
}

# ----------------------------------------------------------
# 2. 禁用安全认证（开发环境）
# ----------------------------------------------------------
$ES_CONFIG = Join-Path $ES_DIR "config\elasticsearch.yml"
$config_content = @"

# === 开发环境配置 ===
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
xpack.security.http.ssl.enabled: false
xpack.security.transport.ssl.enabled: false
"@

Write-Host "[配置] 写入开发环境配置（禁用安全认证）" -ForegroundColor Yellow
# 追加到配置文件末尾
Add-Content -Path $ES_CONFIG -Value $config_content -Encoding UTF8
Write-Host "[完成] 配置已写入" -ForegroundColor Green

# ----------------------------------------------------------
# 3. 启动 ES
# ----------------------------------------------------------
Write-Host ""
Write-Host "[启动] 正在启动 Elasticsearch..." -ForegroundColor Cyan
Write-Host "  ES 将在后台启动，首次启动需要 30-60 秒初始化" -ForegroundColor Gray
Write-Host "  访问 http://localhost:9200 验证是否成功" -ForegroundColor Gray
Write-Host ""

$ES_BIN = Join-Path $ES_DIR "bin\elasticsearch.bat"

# 在新窗口中启动 ES（后台运行）
Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$ES_BIN`" " -WindowStyle Minimized

Write-Host "[等待] 等待 ES 启动（约 30 秒）..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# ----------------------------------------------------------
# 4. 验证连接
# ----------------------------------------------------------
Write-Host "[验证] 检查 ES 连接..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:9200" -Method Get -TimeoutSec 10
    Write-Host "[成功] ES 已启动！" -ForegroundColor Green
    Write-Host "  版本: $($response.version.number)" -ForegroundColor White
    Write-Host "  集群: $($response.cluster_name)" -ForegroundColor White
}
catch {
    Write-Host "[提示] ES 可能仍在初始化中，请稍后手动验证：" -ForegroundColor Yellow
    Write-Host "  curl http://localhost:9200" -ForegroundColor White
}

Write-Host ""
Write-Host "=== 配置完成 ===" -ForegroundColor Cyan
Write-Host "停止 ES: 关闭 elasticsearch.bat 窗口" -ForegroundColor Gray
Write-Host "或运行: taskkill /F /IM java.exe" -ForegroundColor Gray