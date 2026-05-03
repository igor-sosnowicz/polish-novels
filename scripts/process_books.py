import asyncio
from pathlib import Path

from imports_setup import setup_project_imports
setup_project_imports()

from src.processing import LLMProcessor


async def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "books_txt"
    output_dir = project_root / "data" / "book_interactions_data"

    # --- Find resources ---
    book_resources = list(data_dir.rglob("*.txt"))
    if not book_resources:
        print(f"No .txt found in {str(data_dir)}")
        return
    print(f"{len(book_resources)} books found.")

    # --- Processing ---
    llm_processor = LLMProcessor()
    await llm_processor.initialize()

    try:
        for i, book_path in enumerate(book_resources, start=1):
            book_name = book_path.stem  # Skip extension
            
            print(f"\n{'='*50}")
            print(f"[{i}/{len(book_resources)}] Book: {book_name}")
            print(f"Path: {book_path}")
            print(f"{'='*50}")

            book_output_dir = output_dir / book_name
            book_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = book_output_dir / "interactions.json"

            if output_file.exists():
                print(f"-> Results for '{book_name}' already exist. Skipping book.")
                continue

            try:
                book_text = book_path.read_text(encoding="utf-8")
                interactions = await llm_processor.process_book(book_text)

                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(interactions.model_dump_json(indent=4))
                    
                print(f"\n-> Completed successfully! Relations saved in: {output_file}")

            except Exception as e:
                print(f"\n[CRITICAL ERROR] An error occurred while processing the book '{book_name}': {e}")
                # You can add error logging to a file here if you want
    except KeyboardInterrupt:
        print("\nManual interruption (Ctrl+C).")
    except Exception as e:
        print(f"\n[Critical Error] {e}")
    finally:
        await llm_processor.unload_model()

if __name__ == "__main__":
    asyncio.run(main())