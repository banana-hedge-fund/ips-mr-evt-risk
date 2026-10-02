# ips-mr-evt-risk — https://github.com/casey/just

set shell := ["bash", "-euo", "pipefail", "-c"]
set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

default:
	@just --list

# Install Git hooks from lefthook.yaml
setup:
	lefthook install

# Install Python dependencies
restore:
	python3 -m pip install -r requirements.txt

# Byte-compile sources, scripts, and the smoke-test helper
build:
	python3 -m compileall -q src scripts tests

# Same check as build: this repo has no separate debug configuration
build-debug: build

# No auto-formatter is configured. Byte-compile so the hook still fails on syntax errors.
format: build

# Byte-compile; a syntax error fails the recipe
lint: build

# Byte-compile. The full analysis is `just run` and needs data/raw.
test: build

# Download Binance kline archives into data/raw
fetch:
	python3 scripts/fetch_binance.py

# Run the analysis pipeline (results/ )
run:
	python3 -m src.run_all

# Remove bytecode caches
clean:
	find src scripts tests -type d -name __pycache__ -exec rm -rf {} +
