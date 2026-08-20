@echo off
chcp 65001 >nul
echo === Elasticsearch Local Setup ===
echo.

set ES_VERSION=8.15.0
set ES_DIR=data\elasticsearch-%ES_VERSION%
set ES_ZIP=data\elasticsearch.zip
set ES_URL=https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-%ES_VERSION%-windows-x86_64.zip

if exist "%ES_DIR%\bin\elasticsearch.bat" (
    echo [SKIP] ES already exists at %ES_DIR%
    goto :start
)

echo [DOWNLOAD] Downloading ES %ES_VERSION% ...
echo   URL: %ES_URL%
if not exist "data" mkdir data

powershell -Command "& { $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri '%ES_URL%' -OutFile '%ES_ZIP%' }"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Download failed. Please download manually from:
    echo   https://www.elastic.co/downloads/elasticsearch
    echo   Extract to data\elasticsearch-%ES_VERSION%\
    pause
    exit /b 1
)
echo [OK] Download complete

echo [EXTRACT] Extracting ...
powershell -Command "Expand-Archive -Path '%ES_ZIP%' -DestinationPath 'data' -Force"
del "%ES_ZIP%"
echo [OK] Extraction complete

:start
echo [CONFIG] Writing dev config (disable security) ...
echo. >> "%ES_DIR%\config\elasticsearch.yml"
echo # === Dev config === >> "%ES_DIR%\config\elasticsearch.yml"
echo xpack.security.enabled: false >> "%ES_DIR%\config\elasticsearch.yml"
echo xpack.security.enrollment.enabled: false >> "%ES_DIR%\config\elasticsearch.yml"
echo xpack.security.http.ssl.enabled: false >> "%ES_DIR%\config\elasticsearch.yml"
echo xpack.security.transport.ssl.enabled: false >> "%ES_DIR%\config\elasticsearch.yml"
echo [OK] Config written

echo [START] Starting Elasticsearch...
echo   ES will start in a new window. First startup takes 30-60s.
echo   Verify: curl http://localhost:9200
echo.

start "Elasticsearch" /MIN "%ES_DIR%\bin\elasticsearch.bat"

echo [WAIT] Waiting 30 seconds for ES to start...
timeout /t 30 /nobreak >nul

echo [VERIFY] Checking connection...
curl -s http://localhost:9200 2>nul
if %ERRORLEVEL% equ 0 (
    echo [OK] ES is running!
) else (
    echo [INFO] ES may still be initializing. Check manually: curl http://localhost:9200
)

echo.
echo === Setup Complete ===
echo To stop ES: Close the elasticsearch.bat window
echo Or run: taskkill /F /IM java.exe
pause