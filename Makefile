.PHONY: test test-cov lint format install clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make test-cov   - Run tests with coverage"
	@echo "  make lint       - Run linting checks"
	@echo "  make format     - Format code with black and isort"
	@echo "  make clean      - Clean build artifacts"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=pta_anisotropy --cov-report=html --cov-report=term

lint:
	flake8 pta_anisotropy/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 pta_anisotropy/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black pta_anisotropy/ tests/ *.py
	isort pta_anisotropy/ tests/ *.py

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	rm -rf build/ dist/ .eggs/
