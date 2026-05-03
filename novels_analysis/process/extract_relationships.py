"""Script for extracting character relationships from downloaded books."""

import json

from loguru import logger
from tqdm import tqdm

from novels_analysis.config.configuration import BOOKS_DIR, RELATIONSHIPS_DIR
from novels_analysis.process.text_processing import process_book

def main() -> None:
    RELATIONSHIPS_DIR.mkdir(parents=True, exist_ok=True)

    epoch_dirs = sorted(p for p in BOOKS_DIR.iterdir() if p.is_dir())

    if not epoch_dirs:
        logger.error(f"No epoch directories found in {BOOKS_DIR}. Run fetch_dataset.py first.")
        return

    for epoch_dir in tqdm(epoch_dirs, desc="Epochs"):
        epoch = epoch_dir.name
        out_dir = RELATIONSHIPS_DIR / epoch
        out_dir.mkdir(parents=True, exist_ok=True)

        book_files = sorted(epoch_dir.glob("*.txt"))

        for book_path in tqdm(book_files, desc=f"{epoch}", leave=False):
            out_path = out_dir / book_path.with_suffix(".json").name

            if out_path.exists():
                logger.debug(f"Skipping {book_path.name} (already processed).")
                continue

            logger.info(f"Processing {epoch}/{book_path.name} ...")
            txt = book_path.read_text(encoding="utf-8")

            interactions = process_book(txt)

            if not interactions:
                logger.warning(f"No interactions found in {book_path.name}.")

            out_path.write_text(
                json.dumps(interactions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.success(f"Saved {len(interactions)} interactions → {out_path}")


if __name__ == "__main__":
    main()