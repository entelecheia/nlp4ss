SHELL := bash
.DEFAULT_GOAL := help

.PHONY: install
install: ## Create the virtual environment with all dependency groups
	@uv sync

.PHONY: lint
lint: ## Lock consistency and ruff
	@uv lock --locked
	@uv run --only-group lint ruff check .
	@uv run --only-group lint ruff format --check .

.PHONY: test
test: ## Tests with coverage
	@uv run pytest --cov --cov-report=term-missing

.PHONY: verify
verify: lint test ## The one gate: run before reporting any change complete

.PHONY: build
build: ## Build sdist and wheel into dist/
	@uv build

.PHONY: book
book: ## Build the en and ko books (book/_scripts/build.sh)
	@uv run --with-requirements book/requirements.txt bash book/_scripts/build.sh

.PHONY: toc
toc: ## Regenerate the book table of contents
	@uv run python .github/scripts/autotoc.py

symlink-global-docker-env: ## symlink global docker env file for local development
	@DOCKERFILES_SHARE_DIR="$HOME/.local/share/dockerfiles" \
	DOCKER_GLOBAL_ENV_FILENAME=".env.docker" \
	DOCKER_GLOBAL_ENV_FILE="$${DOCKERFILES_SHARE_DIR}/$${DOCKER_GLOBAL_ENV_FILENAME}" \
	[ -f "$${DOCKER_GLOBAL_ENV_FILE}" ] && ln -sf "$${DOCKER_GLOBAL_ENV_FILE}" .env.docker || echo "Global docker env file not found"

docker-login: ## login to docker
	@bash .docker/.docker-scripts/docker-compose.sh login

docker-build: ## build the docker app image
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh build

docker-config: ## show the docker app config
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh config

docker-push: ## push the docker app image
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh push

docker-run: ## run the docker base image
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh run

docker-up: ## launch the docker app image
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh up

docker-up-detach: ## launch the docker app image in detached mode
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh up --detach

docker-tag-latest: ## tag the docker app image as latest
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh tag

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}'
