"""Script for extracting characters and their relationships from texts."""

import asyncio
from pathlib import Path

import pandas as pd
from src.text_processing import extract_relationships, text_into_chunks


async def main() -> None:
    # TODO: Iterate over all books.
    book_file = Path("./data/books_txt/barok/ksiezna_de_cleves.txt")
    book_content = book_file.read_text()
    chunks = text_into_chunks(txt=book_content)

    all_interactions = []
    for chunk in chunks:
        interactions_in_chunk = extract_relationships(chunk)
        all_interactions.extend(interactions_in_chunk)

    df = pd.DataFrame(
        data=all_interactions,
        columns=["character_1", "character_2", "interaction_sentiment"],
    )

    interaction_directory = Path("./data/interactions")
    interaction_directory.mkdir(parents=True, exist_ok=True)
    output_filename = book_file.name.replace(".txt", ".csv")
    df.to_csv(interaction_directory / output_filename)


if __name__ == "__main__":
    asyncio.run(main())
