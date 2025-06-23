# Makefile per ViT-VS Container
# Usa: make build, make run, make test, etc.

# Variabili
IMAGE_NAME = vitqs-standalone
CONTAINER_NAME = vitqs_container
VERSION = latest

# Regole principali
.PHONY: help build run test clean jupyter logs health

help: ## Mostra questo help
	@echo "ViT-VS Container Commands:"
	@echo "=========================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Costruisce l'immagine Docker
	@echo "🔨 Building Docker image..."
	docker build -t $(IMAGE_NAME):$(VERSION) .
	@echo "✅ Build completata!"

run: ## Avvia il container interattivo con X11
	@echo "🚀 Avviando container con X11..."
	@mkdir -p output results
	@if [ -n "$$DISPLAY" ]; then \
		echo "📺 Using X11 display: $$DISPLAY"; \
		docker run -it --rm --gpus all \
			--env DISPLAY=$$DISPLAY \
			--env QT_X11_NO_MITSHM=1 \
			--volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
			--volume ~/.Xauthority:/home/vitqs/.Xauthority:ro \
			--volume $$(pwd)/dataset_small:/home/vitqs/vitqs_app/dataset_small:ro \
			--volume $$(pwd)/output:/home/vitqs/vitqs_app/output:rw \
			--volume $$(pwd)/results:/home/vitqs/vitqs_app/results:rw \
			--name $(CONTAINER_NAME) \
			$(IMAGE_NAME):$(VERSION); \
	else \
		echo "⚠️  No DISPLAY, using headless mode"; \
		make run-headless; \
	fi

run-headless: ## Avvia il container headless (senza display)
	@echo "�️  Avviando container headless..."
	@mkdir -p output results
	docker run -it --rm --gpus all \
		--env MPLBACKEND=Agg \
		--env DISPLAY=:99 \
		--volume $$(pwd)/dataset_small:/home/vitqs/vitqs_app/dataset_small:ro \
		--volume $$(pwd)/output:/home/vitqs/vitqs_app/output:rw \
		--volume $$(pwd)/results:/home/vitqs/vitqs_app/results:rw \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(VERSION) \
		bash -c "Xvfb :99 -screen 0 1024x768x24 & python3 demo.py"

test: ## Esegue test del sistema
	@echo "🧪 Eseguendo test..."
	@mkdir -p output results
	docker run --rm --gpus all \
		-v $$(pwd)/dataset_small:/home/vitqs/vitqs_app/dataset_small:ro \
		-v $$(pwd)/output:/home/vitqs/vitqs_app/output:rw \
		$(IMAGE_NAME):$(VERSION) \
		python3 test_vit.py

health: ## Health check del container
	@echo "🏥 Health check container..."
	docker run --rm --gpus all \
		-v $$(pwd)/dataset_small:/home/vitqs/vitqs_app/dataset_small:ro \
		$(IMAGE_NAME):$(VERSION) \
		python3 container_health_check.py

jupyter: ## Avvia Jupyter Notebook
	@echo "📓 Avviando Jupyter..."
	@mkdir -p output results
	docker run -it --rm --gpus all \
		-v $$(pwd):/home/vitqs/vitqs_app:rw \
		-p 8888:8888 \
		--name vitqs_jupyter \
		$(IMAGE_NAME):$(VERSION) \
		bash -c "pip install jupyter notebook && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"

shell: ## Accesso shell nel container
	@echo "🐚 Accesso shell..."
	docker run -it --rm --gpus all \
		-v $$(pwd):/home/vitqs/vitqs_app:rw \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME):$(VERSION) \
		bash

logs: ## Mostra logs del container
	docker logs $(CONTAINER_NAME)

clean: ## Pulizia immagini e container
	@echo "🧹 Pulizia..."
	-docker container rm $(CONTAINER_NAME)
	-docker image rm $(IMAGE_NAME):$(VERSION)
	docker system prune -f

size: ## Mostra dimensione immagine
	@echo "📏 Dimensione immagine:"
	docker images $(IMAGE_NAME):$(VERSION)

inspect: ## Ispeziona immagine
	@echo "🔍 Dettagli immagine:"
	docker inspect $(IMAGE_NAME):$(VERSION)

# Comandi avanzati
build-no-cache: ## Build senza cache
	docker build --no-cache -t $(IMAGE_NAME):$(VERSION) .

push: ## Push su registry (configura REGISTRY)
	docker tag $(IMAGE_NAME):$(VERSION) $(REGISTRY)/$(IMAGE_NAME):$(VERSION)
	docker push $(REGISTRY)/$(IMAGE_NAME):$(VERSION)

# Sviluppo
dev: ## Ambiente di sviluppo
	docker run -it --rm --gpus all \
		-v $$(pwd):/home/vitqs/vitqs_app:rw \
		-p 8888:8888 \
		--name vitqs_dev \
		$(IMAGE_NAME):$(VERSION) \
		bash

# Info sistema
info: ## Info sistema Docker
	@echo "🖥️  Sistema Docker:"
	docker version
	@echo "\n🎮 GPU Info:"
	-docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

ssh-check: ## Verifica ambiente SSH + X11
	@echo "🔍 SSH Environment Check..."
	python3 ssh_environment_check.py

test-x11: ## Test X11 forwarding
	@echo "🧪 Test X11 forwarding..."
	docker run --rm --gpus all \
		--env DISPLAY=$$DISPLAY \
		--volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
		--volume ~/.Xauthority:/home/vitqs/.Xauthority:ro \
		$(IMAGE_NAME):$(VERSION) \
		xclock
