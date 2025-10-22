# Makefile for ML CLI Development
.PHONY: help install test lint format clean build publish-test publish ci-test

help:
	@echo "ML CLI Development Commands:"
	@echo ""
	@echo "  make install       Install package in development mode"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black and isort"
	@echo "  make clean         Clean build artifacts"
	@echo "  make build         Build distribution packages"
	@echo "  make ci-test       Run full CI pipeline locally"
	@echo "  make publish-test  Publish to TestPyPI"
	@echo "  make publish       Publish to PyPI"
	@echo ""

install:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=ml_cli --cov-report=html --cov-report=term --cov-report=xml

lint:
	@echo "Running flake8 for syntax errors..."
	flake8 ml_cli --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo ""
	@echo "Running flake8 for style issues..."
	flake8 ml_cli --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "Formatting with black..."
	black ml_cli/ tests/
	@echo "Sorting imports with isort..."
	isort ml_cli/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

ci-test: lint test build
	@echo ""
	@echo "✅ CI pipeline completed successfully!"
	@echo ""
	@echo "Checking distribution..."
	twine check dist/*
	@echo ""
	@echo "🎉 All CI steps passed! Ready to publish."

publish-test: build
	@echo "Publishing to TestPyPI..."
	twine upload --repository testpypi dist/*

publish: build
	@echo "Publishing to PyPI..."
	@echo "⚠️  This will publish to the REAL PyPI. Are you sure? [Ctrl+C to cancel]"
	@read -p "Press Enter to continue..."
	twine upload dist/*
