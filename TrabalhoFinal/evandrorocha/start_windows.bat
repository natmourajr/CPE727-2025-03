@echo off
echo ==========================================
echo INICIANDO AMBIENTE - WINDOWS + GPU NVIDIA
echo ==========================================

REM Verificar se nvidia-smi existe
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [AVISO] nvidia-smi nao encontrado!
    echo GPU NVIDIA pode nao estar disponivel.
    echo Continuando automaticamente...
    echo.
)

echo.
echo Detectando GPU NVIDIA...
nvidia-smi --query-gpu=name --format=csv,noheader 2>nul
if %ERRORLEVEL% EQU 0 (
    echo GPU detectada com sucesso!
) else (
    echo [AVISO] Nenhuma GPU detectada. Treinamento sera mais lento.
)

echo.
echo Parando containers existentes...
docker compose down 2>nul

echo.
echo Construindo imagem Docker com suporte a GPU...
set COMPOSE_PROFILES=gpu
docker compose build

if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] Falha ao construir imagem!
    pause
    exit /b 1
)

echo.
echo Iniciando container...
docker compose up -d

if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] Falha ao iniciar container!
    pause
    exit /b 1
)

echo.
echo Aguardando container iniciar...
timeout /t 3 /nobreak >nul

echo.
echo ==========================================
echo Container iniciado com sucesso!
echo ==========================================
echo.
echo JUPYTER LAB: http://localhost:8888
echo.
echo Comandos uteis:
echo   Ver logs:        docker compose logs -f
echo   Parar:          docker compose down
echo   Entrar:         docker compose exec tuberculosis-detection-gpu bash
echo   Status GPU:     docker compose exec tuberculosis-detection-gpu nvidia-smi
echo   Monitorar GPU:  docker compose exec tuberculosis-detection-gpu nvidia-smi -l 1
pause
