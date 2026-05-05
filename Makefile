.RECIPEPREFIX = >

fetch_dataset:
> PYTHONPATH="." uv run scripts/fetch_dataset.py

compress_interactions:
> tar -czvf data/book_interactions_data.tar.gz -C data book_interactions_data book_characters

decompress_interactions:
> tar -xzvf data/book_interactions_data.tar.gz -C data