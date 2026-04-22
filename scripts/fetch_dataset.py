"""Script for fetching a dataset made of plain-text books."""

import asyncio
from pathlib import Path

from tqdm import tqdm

from src.api import fetch_book, fetch_epochs, fetch_prose, to_filename
from src.configuration import MAX_BOOKS_PER_EPOCH


async def main() -> None:
    epochs = await fetch_epochs()
    books_per_epoch = await fetch_prose(epochs=epochs)
    for epoch in tqdm(epochs, desc="epochs"):
        books_in_epoch = books_per_epoch[epoch]
        epoch_directory = Path("./data") / epoch
        epoch_directory.mkdir(parents=True, exist_ok=True)
        saved_books = len(list(epoch_directory.glob("*.txt")))

        if saved_books >= MAX_BOOKS_PER_EPOCH:
            continue

        for book_href in tqdm(books_in_epoch, desc=f"{epoch} books", leave=False):
            if saved_books >= MAX_BOOKS_PER_EPOCH:
                break

            filepath = epoch_directory / to_filename(book_href)
            if filepath.exists():
                continue

            book_txt = await fetch_book(book_href)
            if book_txt:
                filepath.write_text(book_txt)
                saved_books += 1


if __name__ == "__main__":
    asyncio.run(main())
