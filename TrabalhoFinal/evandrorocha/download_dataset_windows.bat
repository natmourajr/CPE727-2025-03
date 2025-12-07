@echo off
echo ==========================================
echo DOWNLOAD DO DATASET SHENZHEN - WINDOWS
echo ==========================================

echo.
echo Construindo container...
set COMPOSE_PROFILES=gpu
docker compose build

if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] Falha ao construir container!
    pause
    exit /b 1
)

echo.
echo Iniciando download do dataset...
echo [AVISO] O download automatico pode falhar devido a restricoes do site NIH.
echo.
docker compose run --rm tuberculosis-detection-gpu python src/download_data.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==========================================
    echo DOWNLOAD AUTOMATICO FALHOU
    echo ==========================================
    echo.
    echo INSTRUCOES PARA DOWNLOAD MANUAL:
    echo.
    echo 1. Acesse o site oficial:
    echo    https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#tuberculosis-image-data-sets
    echo.
    echo 2. Localize 'Shenzhen Hospital X-ray Set' e clique em Download
    echo.
    echo 3. Baixe o arquivo ChinaSet_AllFiles.zip (aprox. 440 MB^)
    echo.
    echo 4. Coloque o arquivo em:
    echo    %CD%\data\shenzhen_dataset.zip
    echo.
    echo 5. Execute novamente este script
    echo.
    pause
    exit /b 1
)

echo.
echo Verificando dataset...
docker compose run --rm tuberculosis-detection-gpu python src/download_data.py --verify-only

echo.
echo ==========================================
echo Dataset pronto para uso!
echo ==========================================
echo.
echo Proximos passos:
echo   1. Inicie o ambiente: start_windows.bat
echo   2. Acesse o Jupyter: http://localhost:8888
echo   3. Ou treine modelos: train_all_windows.bat
echo.
pause
