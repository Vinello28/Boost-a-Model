#!/bin/bash

# Script per build e run del container ViT-VS

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 ViT-VS Docker Setup${NC}"
echo "=================================="

# Check se Docker è installato
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker non trovato. Installa Docker prima di continuare.${NC}"
    exit 1
fi

# Check se nvidia-docker è disponibile (per GPU)
if command -v nvidia-docker &> /dev/null || docker info | grep -q "nvidia"; then
    echo -e "${GREEN}✅ GPU support disponibile${NC}"
    GPU_SUPPORT=true
else
    echo -e "${YELLOW}⚠️  GPU support non disponibile. Il container userà solo CPU.${NC}"
    GPU_SUPPORT=false
fi

# Funzione per build
build_container() {
    echo -e "${BLUE}🔨 Building Docker image...${NC}"
    docker build -t vitqs-standalone:latest .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Build completata con successo!${NC}"
    else
        echo -e "${RED}❌ Errore durante il build${NC}"
        exit 1
    fi
}

# Funzione per run del container con X11
run_container() {
    echo -e "${BLUE}🚀 Avviando container ViT-VS...${NC}"
    
    # Crea directory output se non esistono
    mkdir -p output results
    
    # Check se X11 è disponibile
    if [ -n "$DISPLAY" ]; then
        echo -e "${GREEN}📺 X11 Display disponibile: $DISPLAY${NC}"
        X11_ARGS="--env DISPLAY=$DISPLAY 
                  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw 
                  --volume $HOME/.Xauthority:/home/vitqs/.Xauthority:ro"
    else
        echo -e "${YELLOW}⚠️  X11 non disponibile, usando Xvfb virtuale${NC}"
        X11_ARGS="--env DISPLAY=:99"
    fi
    
    if [ "$GPU_SUPPORT" = true ]; then
        docker run -it --rm --gpus all \
            $X11_ARGS \
            --env QT_X11_NO_MITSHM=1 \
            --env MPLBACKEND=Agg \
            -v $(pwd)/dataset_small:/home/vitqs/vitqs_app/dataset_small:ro \
            -v $(pwd)/output:/home/vitqs/vitqs_app/output:rw \
            -v $(pwd)/results:/home/vitqs/vitqs_app/results:rw \
            --name vitqs_container \
            vitqs-standalone:latest
    else
        docker run -it --rm \
            $X11_ARGS \
            --env QT_X11_NO_MITSHM=1 \
            --env MPLBACKEND=Agg \
            -v $(pwd)/dataset_small:/home/vitqs/vitqs_app/dataset_small:ro \
            -v $(pwd)/output:/home/vitqs/vitqs_app/output:rw \
            -v $(pwd)/results:/home/vitqs/vitqs_app/results:rw \
            --name vitqs_container \
            vitqs-standalone:latest
    fi
}

# Funzione per run con Jupyter
run_jupyter() {
    echo -e "${BLUE}📓 Avviando Jupyter Notebook...${NC}"
    
    mkdir -p output results
    
    if [ "$GPU_SUPPORT" = true ]; then
        docker run -it --rm --gpus all \
            -v $(pwd):/home/vitqs/vitqs_app:rw \
            -p 8888:8888 \
            --name vitqs_jupyter \
            vitqs-standalone:latest \
            bash -c "pip install jupyter notebook && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"
    else
        docker run -it --rm \
            -v $(pwd):/home/vitqs/vitqs_app:rw \
            -p 8888:8888 \
            --name vitqs_jupyter \
            vitqs-standalone:latest \
            bash -c "pip install jupyter notebook && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"
    fi
}

# Funzione per test
run_test() {
    echo -e "${BLUE}🧪 Eseguendo test ViT...${NC}"
    
    mkdir -p output results
    
    if [ "$GPU_SUPPORT" = true ]; then
        docker run --rm --gpus all \
            -v $(pwd)/dataset_small:/home/vitqs/vitqs_app/dataset_small:ro \
            -v $(pwd)/output:/home/vitqs/vitqs_app/output:rw \
            vitqs-standalone:latest \
            python3 test_vit.py
    else
        docker run --rm \
            -v $(pwd)/dataset_small:/home/vitqs/vitqs_app/dataset_small:ro \
            -v $(pwd)/output:/home/vitqs/vitqs_app/output:rw \
            vitqs-standalone:latest \
            python3 test_vit.py
    fi
}

# Funzione per run headless (senza display)
run_headless() {
    echo -e "${BLUE}🖥️  Avviando container headless...${NC}"
    
    mkdir -p output results
    
    if [ "$GPU_SUPPORT" = true ]; then
        docker run -it --rm --gpus all \
            --env MPLBACKEND=Agg \
            --env DISPLAY=:99 \
            -v $(pwd)/dataset_small:/home/vitqs/vitqs_app/dataset_small:ro \
            -v $(pwd)/output:/home/vitqs/vitqs_app/output:rw \
            -v $(pwd)/results:/home/vitqs/vitqs_app/results:rw \
            --name vitqs_headless \
            vitqs-standalone:latest \
            bash -c "Xvfb :99 -screen 0 1024x768x24 & python3 demo.py"
    else
        docker run -it --rm \
            --env MPLBACKEND=Agg \
            --env DISPLAY=:99 \
            -v $(pwd)/dataset_small:/home/vitqs/vitqs_app/dataset_small:ro \
            -v $(pwd)/output:/home/vitqs/vitqs_app/output:rw \
            -v $(pwd)/results:/home/vitqs/vitqs_app/results:rw \
            --name vitqs_headless \
            vitqs-standalone:latest \
            bash -c "Xvfb :99 -screen 0 1024x768x24 & python3 demo.py"
    fi
}

# Funzione per testare X11
test_x11() {
    echo -e "${BLUE}🔍 Test X11 forwarding...${NC}"
    
    if [ "$GPU_SUPPORT" = true ]; then
        docker run --rm --gpus all \
            --env DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            -v $HOME/.Xauthority:/home/vitqs/.Xauthority:ro \
            vitqs-standalone:latest \
            xclock
    else
        docker run --rm \
            --env DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            -v $HOME/.Xauthority:/home/vitqs/.Xauthority:ro \
            vitqs-standalone:latest \
            xclock
    fi
}

# Parse arguments
case "$1" in
    "build")
        build_container
        ;;
    "run")
        run_container
        ;;
    "jupyter")
        run_jupyter
        ;;
    "test")
        run_test
        ;;
    "headless")
        run_headless
        ;;
    "x11test")
        test_x11
        ;;
    "all")
        build_container
        run_container
        ;;
    *)
        echo -e "${YELLOW}Uso: $0 {build|run|jupyter|test|headless|x11test|all}${NC}"
        echo ""
        echo -e "${BLUE}Comandi disponibili:${NC}"
        echo "  build     - Costruisce l'immagine Docker"
        echo "  run       - Avvia il container interattivo"
        echo "  jupyter   - Avvia Jupyter Notebook (porta 8888)"
        echo "  test      - Esegue test ViT"
        echo "  headless   - Avvia il container in modalità headless"
        echo "  x11test   - Testa il forwarding X11"
        echo "  all       - Build + Run"
        echo ""
        echo -e "${YELLOW}Esempi:${NC}"
        echo "  ./docker_setup.sh build"
        echo "  ./docker_setup.sh run"
        echo "  ./docker_setup.sh jupyter"
        exit 1
        ;;
esac
