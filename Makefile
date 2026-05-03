fetch_dataset:
	PYTHONPATH="." uv run scripts/fetch_dataset.py

# Full pipeline
run_all: fetch extract build analyze

# Individual steps
fetch:
	@echo "==> [1/4] Fetching books..."
	uv run python -m novels_analysis.fetch.fetch_dataset

extract:
	@echo "==> [2/4] Extracting character relationships via Gemini..."
	uv run python -m novels_analysis.process.extract_relationships

build:
	@echo "==> [3/4] Building NetworkX graphs from relationships..."
	uv run python -m novels_analysis.graph.build_graphs

analyze:
	@echo "==> [4/4] Computing graph features -> data/features.csv..."
	uv run python -m novels_analysis.graph.analyze_graphs

clean:
	@echo "Removing generated data..."
	uv run python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['data/relationships','data/graphs']]; import os; os.path.exists('data/features.csv') and os.remove('data/features.csv')"

help:
	@echo ""
	@echo "  make fetch      - Download books from wolnelektury.pl"
	@echo "  make extract    - Extract relationships (needs GEMINI_API_KEY in .env)"
	@echo "  make build      - Build GraphML graph files"
	@echo "  make analyze    - Compute features -> data/features.csv"
	@echo "  make run_all    - Run full pipeline"
	@echo "  make clean      - Remove generated data (keeps books)"
	@echo ""