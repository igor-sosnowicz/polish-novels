fetch_dataset:
	PYTHONPATH="." uv run scripts/fetch_dataset.py


compress_interactions:
	tar -czvf data/book_interactions_data.tar.gz data/book_interactions_data

decompress_interactions:
	tar -xzvf data/book_interactions_data.tar.gz -C data/