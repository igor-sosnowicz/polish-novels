"""Script for fetching a dataset made of plain-text books."""

import asyncio
from pathlib import Path
from src.api import fetch_book, fetch_epochs, fetch_prose, to_filename


async def main() -> None:
    epochs = await fetch_epochs()
    books_per_epoch = await fetch_prose(limit=5, epochs=epochs)
    for epoch, books_in_epoch in books_per_epoch.items():
        epoch_directory = Path("./data") / epoch
        epoch_directory.mkdir(parents=True, exist_ok=True)

        for book_href in books_in_epoch:
            filepath = epoch_directory / to_filename(book_href)
            if filepath.exists():
                continue

            book_txt = await fetch_book(book_href)
            if book_txt:
                filepath.write_text(book_txt)


if __name__ == "__main__":
    asyncio.run(main())
