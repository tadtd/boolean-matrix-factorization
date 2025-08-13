# Detect operating system
ifeq ($(OS),Windows_NT)
	# Windows paths
	VENV_BIN = $(VENV_NAME)/Scripts
	ACTIVATE_CMD = $(VENV_NAME)\Scripts\activate.bat
	RM_CMD = rmdir /s /q
	MKDIR_CMD = if not exist
	EXIST_CMD = if exist
else
	# Unix-like systems (Linux, macOS)
	VENV_BIN = $(VENV_NAME)/bin
	ACTIVATE_CMD = source $(VENV_NAME)/bin/activate
	RM_CMD = rm -rf
	MKDIR_CMD = test -d
	EXIST_CMD = test -d
endif

# Virtual environment settings
VENV_NAME = venv
PYTHON = python3
PIP = $(VENV_BIN)/pip
PYTHON_VENV = $(VENV_BIN)/python

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup     - Create virtual environment and install dependencies"
	@echo "  venv      - Create virtual environment only"
	@echo "  install   - Install dependencies in existing virtual environment"
	@echo "  activate  - Activate virtual environment"
	@echo "  test      - Run tests"
	@echo "  run       - Run the main application"
	@echo "  clean     - Remove virtual environment"
	@echo "  status    - Show virtual environment status"
	@echo "  help      - Show this help message"

# Create virtual environment
.PHONY: venv
venv:
ifeq ($(OS),Windows_NT)
	@if not exist $(VENV_NAME) ( \
		echo Creating virtual environment... && \
		$(PYTHON) -m venv $(VENV_NAME) \
	) else ( \
		echo Virtual environment already exists. \
	)
else
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_NAME); \
	else \
		echo "Virtual environment already exists."; \
	fi
endif

# Install dependencies
.PHONY: install
install: venv
	@echo Installing dependencies...
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

# Complete setup (create venv + install dependencies)
.PHONY: setup
setup: venv install
	@echo Setup complete!
	@echo To activate the virtual environment, run:
	@echo   $(ACTIVATE_CMD)

# Activate virtual environment
.PHONY: activate
activate: venv
ifeq ($(OS),Windows_NT)
	@$(VENV_NAME)\Scripts\activate.bat
else
	@echo "Activating virtual environment..."
	@exec bash -c "source $(VENV_NAME)/bin/activate; exec bash"
endif

# Run tests
.PHONY: test
test: venv
	@echo Running tests...
	@$(PYTHON_VENV) test.py

# Run main application
.PHONY: run
run: venv
	@echo Running application...
	@$(PYTHON_VENV) -m BMF

# Clean up virtual environment
.PHONY: clean
clean:
ifeq ($(OS),Windows_NT)
	@if exist $(VENV_NAME) ( \
		echo Removing virtual environment... && \
		rmdir /s /q $(VENV_NAME) \
	) else ( \
		echo No virtual environment to remove. \
	)
else
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "Removing virtual environment..."; \
		rm -rf $(VENV_NAME); \
	else \
		echo "No virtual environment to remove."; \
	fi
endif

# Development targets
.PHONY: dev-install
dev-install: venv
	@echo Installing development dependencies...
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@$(PIP) install pytest flake8 black

# Format code
.PHONY: format
format: venv
	@echo Formatting code...
	@$(PYTHON_VENV) -m black BMF/ problems/ test.py

# Lint code
.PHONY: lint
lint: venv
	@echo Linting code...
	@$(PYTHON_VENV) -m flake8 BMF/ problems/ test.py

# Show virtual environment status
.PHONY: status
status:
ifeq ($(OS),Windows_NT)
	@if exist $(VENV_NAME) ( \
		echo Virtual environment exists at: $(VENV_NAME) && \
		echo Python executable: $(PYTHON_VENV) && \
		$(PYTHON_VENV) --version \
	) else ( \
		echo No virtual environment found. Run 'make setup' to create one. \
	)
else
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "Virtual environment exists at: $(VENV_NAME)"; \
		echo "Python executable: $(PYTHON_VENV)"; \
		$(PYTHON_VENV) --version; \
	else \
		echo "No virtual environment found. Run 'make setup' to create one."; \
	fi
endif