@echo off
REM Script Windows per setup Docker ViT-VS

echo 🐳 ViT-VS Docker Setup
echo ==================================

REM Check se Docker è installato
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker non trovato. Installa Docker Desktop prima di continuare.
    pause
    exit /b 1
)

REM Check GPU support
docker info | findstr nvidia >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ GPU support disponibile
    set GPU_SUPPORT=true
) else (
    echo ⚠️  GPU support non disponibile. Il container userà solo CPU.
    set GPU_SUPPORT=false
)

if "%1"=="build" goto build
if "%1"=="run" goto run
if "%1"=="headless" goto headless
if "%1"=="jupyter" goto jupyter
if "%1"=="test" goto test
if "%1"=="ssh-check" goto ssh-check
if "%1"=="all" goto all
goto usage

:build
echo 🔨 Building Docker image...
docker build -t vitqs-standalone:latest .
if %errorlevel% equ 0 (
    echo ✅ Build completata con successo!
) else (
    echo ❌ Errore durante il build
    exit /b 1
)
goto end

:run
echo 🚀 Avviando container ViT-VS...
if not exist output mkdir output
if not exist results mkdir results

REM Configura ambiente per Windows/WSL
set "X11_ARGS="
if defined DISPLAY (
    echo 📺 X11 Display disponibile: %DISPLAY%
    set "X11_ARGS=-e DISPLAY=%DISPLAY% -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
) else (
    echo ⚠️  X11 non disponibile, usando backend Agg
    set "X11_ARGS=-e MPLBACKEND=Agg"
)

if "%GPU_SUPPORT%"=="true" (
    docker run -it --rm --gpus all ^
        %X11_ARGS% ^
        -e QT_X11_NO_MITSHM=1 ^
        -v "%cd%\dataset_small:/home/vitqs/vitqs_app/dataset_small:ro" ^
        -v "%cd%\output:/home/vitqs/vitqs_app/output:rw" ^
        -v "%cd%\results:/home/vitqs/vitqs_app/results:rw" ^
        --name vitqs_container ^
        vitqs-standalone:latest
) else (
    docker run -it --rm ^
        %X11_ARGS% ^
        -e QT_X11_NO_MITSHM=1 ^
        -v "%cd%\dataset_small:/home/vitqs/vitqs_app/dataset_small:ro" ^
        -v "%cd%\output:/home/vitqs/vitqs_app/output:rw" ^
        -v "%cd%\results:/home/vitqs/vitqs_app/results:rw" ^
        --name vitqs_container ^
        vitqs-standalone:latest
)
goto end

:headless
echo 🖥️  Avviando container headless...
if not exist output mkdir output
if not exist results mkdir results

if "%GPU_SUPPORT%"=="true" (
    docker run -it --rm --gpus all ^
        -e MPLBACKEND=Agg ^
        -e DISPLAY=:99 ^
        -v "%cd%\dataset_small:/home/vitqs/vitqs_app/dataset_small:ro" ^
        -v "%cd%\output:/home/vitqs/vitqs_app/output:rw" ^
        -v "%cd%\results:/home/vitqs/vitqs_app/results:rw" ^
        --name vitqs_headless ^
        vitqs-standalone:latest ^
        bash -c "Xvfb :99 -screen 0 1024x768x24 & python3 demo.py"
) else (
    docker run -it --rm ^
        -e MPLBACKEND=Agg ^
        -e DISPLAY=:99 ^
        -v "%cd%\dataset_small:/home/vitqs/vitqs_app/dataset_small:ro" ^
        -v "%cd%\output:/home/vitqs/vitqs_app/output:rw" ^
        -v "%cd%\results:/home/vitqs/vitqs_app/results:rw" ^
        --name vitqs_headless ^
        vitqs-standalone:latest ^
        bash -c "Xvfb :99 -screen 0 1024x768x24 & python3 demo.py"
)
goto end

:ssh-check
echo 🔍 SSH Environment Check...
python ssh_environment_check.py
goto end

:jupyter
echo 📓 Avviando Jupyter Notebook...
if not exist output mkdir output
if not exist results mkdir results

if "%GPU_SUPPORT%"=="true" (
    docker run -it --rm --gpus all ^
        -v "%cd%:/home/vitqs/vitqs_app:rw" ^
        -p 8888:8888 ^
        --name vitqs_jupyter ^
        vitqs-standalone:latest ^
        bash -c "pip install jupyter notebook && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"
) else (
    docker run -it --rm ^
        -v "%cd%:/home/vitqs/vitqs_app:rw" ^
        -p 8888:8888 ^
        --name vitqs_jupyter ^
        vitqs-standalone:latest ^
        bash -c "pip install jupyter notebook && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"
)
echo Jupyter disponibile su: http://localhost:8888
goto end

:test
echo 🧪 Eseguendo test ViT...
if not exist output mkdir output
if not exist results mkdir results

if "%GPU_SUPPORT%"=="true" (
    docker run --rm --gpus all ^
        -v "%cd%\dataset_small:/home/vitqs/vitqs_app/dataset_small:ro" ^
        -v "%cd%\output:/home/vitqs/vitqs_app/output:rw" ^
        vitqs-standalone:latest ^
        python3 test_vit.py
) else (
    docker run --rm ^
        -v "%cd%\dataset_small:/home/vitqs/vitqs_app/dataset_small:ro" ^
        -v "%cd%\output:/home/vitqs/vitqs_app/output:rw" ^
        vitqs-standalone:latest ^
        python3 test_vit.py
)
goto end

:all
call :build
if %errorlevel% equ 0 call :run
goto end

:usage
echo Uso: %0 {build^|run^|headless^|jupyter^|test^|ssh-check^|all}
echo.
echo Comandi disponibili:
echo   build     - Costruisce l'immagine Docker
echo   run       - Avvia il container con X11 support
echo   headless  - Avvia il container senza display
echo   jupyter   - Avvia Jupyter Notebook (porta 8888)
echo   test      - Esegue test ViT
echo   ssh-check - Verifica ambiente SSH
echo   all       - Build + Run
echo.
echo Esempi SSH:
echo   docker_setup.bat ssh-check
echo   docker_setup.bat headless
echo   docker_setup.bat run
pause
exit /b 1

:end
