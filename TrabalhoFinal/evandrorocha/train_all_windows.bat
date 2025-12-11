@echo off
echo ==========================================
echo TREINAMENTO DE TODOS OS MODELOS - GPU
echo ==========================================

REM Verificar se container esta rodando
docker compose ps | findstr "tuberculosis-detection-gpu" | findstr "Up" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] Container nao esta rodando!
    echo Execute primeiro: start_windows.bat
    pause
    exit /b 1
)

echo.
echo Verificando GPU...
docker compose exec tuberculosis-detection-gpu nvidia-smi --query-gpu=name --format=csv,noheader
echo.

REM Lista de modelos para treinar
set MODELS=resnet50 densenet121 efficientnet_b0

REM Batch size otimizado para GPU
set BATCH_SIZE=16
set EPOCHS=50

echo Configuracao de treinamento:
echo   Modelos: %MODELS%
echo   Batch size: %BATCH_SIZE%
echo   Epochs: %EPOCHS%
echo.
echo Pressione qualquer tecla para iniciar o treinamento...
pause >nul

for %%m in (%MODELS%) do (
    echo.
    echo ==========================================
    echo Treinando modelo: %%m
    echo ==========================================
    echo Inicio: %date% %time%
    echo.
    
    docker compose exec tuberculosis-detection-gpu python src/train.py ^
        --model %%m ^
        --epochs %EPOCHS% ^
        --batch-size %BATCH_SIZE%
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERRO] Falha ao treinar modelo %%m
        pause
        exit /b 1
    )
    
    echo.
    echo Modelo %%m treinado com sucesso!
    echo Fim: %date% %time%
)

echo.
echo ==========================================
echo TODOS OS MODELOS TREINADOS!
echo ==========================================
echo.
echo Proximos passos:
echo   Avaliar modelos: docker compose exec tuberculosis-detection-gpu python src/evaluate.py
echo   Ver resultados:  explorer .\results
echo.
pause
