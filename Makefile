# Makefile for Text-to-Graph Project
# -------------------------------
# Commands for development, testing, and deployment

# Project configuration
PROJECT_NAME := text-to-graph
DOCKER_COMPOSE := docker compose

# Color definitions
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

# Help text
TARGET_MAX_CHAR_NUM=20

## Show help
display_help:
	@echo ''
	${YELLOW}Text-to-Graph Project${RESET}
	----------------------
	${GREEN}Build:${RESET}
	  ${YELLOW}make build${RESET}		Build all services
	  ${YELLOW}make build-<service>${RESET}	Build a specific service

	${GREEN}Development:${RESET}
	  ${YELLOW}make up${RESET}			Build (if needed) and start all services
	  ${YELLOW}make start${RESET}		Start existing containers without building
	  ${YELLOW}make rebuild${RESET}		Force rebuild and restart all services
	  ${YELLOW}make down${RESET}		Stop and remove all containers
	  ${YELLOW}make logs${RESET}		View logs from all services
	  ${YELLOW}make logs <service>${RESET}	View logs from a specific service

	${GREEN}Testing:${RESET}
	  ${YELLOW}make test${RESET}		Run tests
	  ${YELLOW}make test-file f=path/to/test.py${RESET}  Run a specific test file

	${GREEN}Maintenance:${RESET}
	  ${YELLOW}make clean${RESET}		Remove all containers and volumes
	  ${YELLOW}make prune${RESET}		Remove unused Docker resources
	  ${YELLOW}make requirements${RESET}	Update requirements.txt

	${GREEN}AutoGOAL:${RESET}
	  ${YELLOW}make autogoal-shell${RESET}	Open a shell in the AutoGOAL container
	  ${YELLOW}make autogoal-test${RESET}	Run AutoGOAL tests

	${GREEN}Utility:${RESET}
	  ${YELLOW}make format${RESET}		Format code with Black
	  ${YELLOW}make lint${RESET}		Run linters
	  ${YELLOW}make typecheck${RESET}	Run type checking

.PHONY: help
display_help help:
	@awk '\
	/^### /{gsub(/### /, "");print "\n"$$1"\n"} \
	/^## /{gsub(/## /, "");print "\n"$$1":\n"} \
	/^[\t ].*## /{gsub(/^[\t ]*[^:]*:[\t ]*## /, "");print "  "$$0}' $(MAKEFILE_LIST)

# Build
## Build all services
build:
	@echo "${GREEN}🔨 Building all services...${RESET}"
	${DOCKER_COMPOSE} build --pull


build-no-cache:
	@echo "${GREEN}🔨 Building all services...${RESET}"
	${DOCKER_COMPOSE} build --no-cache --pull

## Build a specific service
build-%:
	@echo "${GREEN}🔨 Building $* service...${RESET}"
	${DOCKER_COMPOSE} build --no-cache --pull $*

# Development
## Start all services (build if needed)
up:
	@echo "${GREEN}🚀 Starting all services...${RESET}"
	${DOCKER_COMPOSE} up -d

## Start services without building
start:
	@echo "${GREEN}🚀 Starting existing containers...${RESET}"
	${DOCKER_COMPOSE} up -d

## Rebuild and restart services
rebuild: build up

## Stop all services
down:
	@echo "${YELLOW}🛑 Stopping all services...${RESET}"
	${DOCKER_COMPOSE} down

## View logs
logs:
	${DOCKER_COMPOSE} logs -f $(filter-out $@,$(MAKECMDGOALS))

# Testing
## Run all tests
test:
	@echo "${GREEN}🧪 Running tests...${RESET}"
	${DOCKER_COMPOSE} exec app pytest tests/ -v

## Run a specific test file
test-file:
	@if [ -z "$(f)" ]; then \
		echo "${YELLOW}Please specify a test file with f=path/to/test.py${RESET}"; \
		exit 1; \
	fi
	@echo "${GREEN}🧪 Running test file: $(f)${RESET}"
	${DOCKER_COMPOSE} exec app pytest $(f) -v

# Maintenance
## Remove all containers and volumes
clean:
	@echo "${YELLOW}🧹 Cleaning up...${RESET}"
	${DOCKER_COMPOSE} down -v --remove-orphans --rmi all

## Remove unused Docker resources
prune:
	@echo "${YELLOW}🧹 Pruning Docker resources...${RESET}"
	docker system prune -a --volumes

## Update requirements.txt
requirements:
	@echo "${GREEN}📦 Updating requirements.txt...${RESET}"
	docker compose exec app poetry export -f requirements.txt --output requirements.txt --without-hashes

# AutoGOAL
## Open a shell in the AutoGOAL container
autogoal-shell:
	${DOCKER_COMPOSE} exec -u coder autogoal bash

## Run AutoGOAL tests
autogoal-test:
	@echo "${GREEN}🧪 Running AutoGOAL tests...${RESET}"
	${DOCKER_COMPOSE} exec -u coder autogoal make test

# GraphSVX
## Open a shell in the GraphSVX container
graphsvx-shell:
	${DOCKER_COMPOSE} exec -u root graphsvx bash

## Run GraphSVX tests
graphsvx-test:
	@echo "${GREEN}🧪 Running GraphSVX tests...${RESET}"
	${DOCKER_COMPOSE} exec graphsvx pytest -v || echo "No tests found or pytest not installed in GraphSVX container."

# SubgraphX
## Open a shell in the SubgraphX container
subgraphx-shell:
	${DOCKER_COMPOSE} exec subgraphx bash

# TokenSHAP
## Open a shell in the TokenSHAP container
tokenshap-shell:
	${DOCKER_COMPOSE} exec tokenshap bash

# Utility
## Format code with Black
format:
	@echo "${GREEN}🎨 Formatting code...${RESET}"
	docker compose exec app black .

## Run linters
lint:
	@echo "${GREEN}🔍 Running linters...${RESET}"
	docker compose exec app flake8 .

## Run type checking
typecheck:
	@echo "${GREEN}🔍 Running type checking...${RESET}"
	docker compose exec app mypy .

# =============================================================================
# Paper Reproduction Pipeline
# =============================================================================
# Run complete paper methodology (Sections 3.1-3.6)

## Run full paper reproduction pipeline
reproduce:
	@echo "${GREEN}🚀 Running Full Paper Reproduction Pipeline${RESET}"
	@echo "Step 1/6: Fine-tuning LLM (Section 3.2)..."
	${DOCKER_COMPOSE} exec app bash scripts/01_finetune_llm.sh
	@echo "Step 2/6: Building graphs (Section 3.1)..."
	${DOCKER_COMPOSE} exec app bash scripts/02_build_graphs.sh
	@echo "Step 3/6: Generating embeddings (Section 3.2)..."
	${DOCKER_COMPOSE} exec app bash scripts/03_generate_embeddings.sh
	@echo "Step 4/6: Training GNNs (Section 3.3)..."
	${DOCKER_COMPOSE} exec app bash scripts/04_train_gnns.sh
	@echo "Step 5/6: Running explainability (Section 3.4)..."
	bash scripts/05_run_explainers.sh
	@echo "Step 6/6: Running analytics (Section 3.5-3.6)..."
	${DOCKER_COMPOSE} exec app bash scripts/06_run_analytics.sh
	@echo "${GREEN}✅ Pipeline complete!${RESET}"

## Step 1: Fine-tune LLM
step-1-finetune:
	@echo "${GREEN}Step 1: Fine-tuning LLM...${RESET}"
	${DOCKER_COMPOSE} exec app bash scripts/01_finetune_llm.sh

## Step 2: Build graphs
step-2-graphs:
	@echo "${GREEN}Step 2: Building graphs...${RESET}"
	${DOCKER_COMPOSE} exec app bash scripts/02_build_graphs.sh

## Step 3: Generate embeddings
step-3-embeddings:
	@echo "${GREEN}Step 3: Generating embeddings...${RESET}"
	${DOCKER_COMPOSE} exec app bash scripts/03_generate_embeddings.sh

## Step 4: Train GNNs
step-4-train:
	@echo "${GREEN}Step 4: Training GNNs...${RESET}"
	${DOCKER_COMPOSE} exec app bash scripts/04_train_gnns.sh

## Step 5: Run explainability
step-5-explain:
	@echo "${GREEN}Step 5: Running explainability...${RESET}"
	bash scripts/05_run_explainers.sh

## Step 6: Run analytics
step-6-analytics:
	@echo "${GREEN}Step 6: Running analytics...${RESET}"
	${DOCKER_COMPOSE} exec app bash scripts/06_run_analytics.sh

# Handle arguments with spaces
%:
	@:

