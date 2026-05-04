import json
import glob
from pathlib import Path
from typing import cast
from colorama import Fore

from imports_setup import setup_project_imports
setup_project_imports()

from src.models import InteractionList, CharacterMapping

def generate_characters_summaries() -> None:
    interactions_files: list[str] = glob.glob("data/book_interactions_data/*/interactions.json", recursive=True)
    book_characters_dir = Path("data/book_characters")

    for interactions_file_path in interactions_files:
        book_title = Path(interactions_file_path).parent.name
        output_dir = book_characters_dir / book_title
        output_dir.mkdir(parents=True, exist_ok=True)

        print(Fore.GREEN + f"\nProcessing book: {book_title}" + Fore.RESET)
        with open(interactions_file_path, "r", encoding="utf-8") as interactions_file:
            interactions_data = json.load(interactions_file)
            
            characters = set()
            for interaction in interactions_data.get("interactions", []):
                characters.add(interaction.get("character_1", ""))
                characters.add(interaction.get("character_2", ""))

            characters.discard("")
            characters_data = {
                "book_title": book_title,
                "characters": list(characters)
            }
            with open(output_dir / "characters.json", "w", encoding="utf-8") as characters_file:
                json.dump(characters_data, characters_file, ensure_ascii=False, indent=4)
        print(Fore.GREEN + f"Characters extracted and saved for '{book_title}'." + Fore.RESET)


def _merge_with_mapping(interactions_file_path: str, mapping_file_path: str) -> InteractionList:
    with open(interactions_file_path, "r", encoding="utf-8") as interactions_file:
        interactions = InteractionList.model_validate_json(interactions_file.read())

    with open(mapping_file_path, "r", encoding="utf-8") as mapping_file:
        character_mapping = CharacterMapping.model_validate_json(mapping_file.read())

    interactions_to_drop: InteractionList = InteractionList(interactions=[])
    for interaction in interactions.interactions:
        if interaction.character_1 in character_mapping.drop_or_non_character or interaction.character_2 in character_mapping.drop_or_non_character:
            interactions_to_drop.interactions.append(interaction)
            continue

        if any([
            interaction.character_1 not in character_mapping.canonical_characters,
            interaction.character_2 not in character_mapping.canonical_characters,
            interaction.character_1 not in character_mapping.canonical_non_human_personifiable_characters,
            interaction.character_2 not in character_mapping.canonical_non_human_personifiable_characters,
        ]):
            mapped_characters = (
                character_mapping.alias_to_canonical.get(interaction.character_1, None),
                character_mapping.alias_to_canonical.get(interaction.character_2, None),
            )
            if all(mapped_characters):
                interaction.character_1, interaction.character_2 = cast(tuple[str, str], mapped_characters)
            else:
                interactions_to_drop.interactions.append(interaction)
                continue

    print("Total interactions:", len(interactions.interactions))
    print("Interactions to drop:", len(interactions_to_drop.interactions))
    interactions = interactions - interactions_to_drop
    print("Total interactions (after merging):", len(interactions.interactions))
    return interactions
    

def merge_characters() -> None:
    character_mappings_paths = glob.glob("data/book_characters/*/character_mapping.json", recursive=True)
    character_mapping: list[tuple[str, str]] = [
        (Path(path).parent.name, path) for path in character_mappings_paths
    ]

    interactions_files: list[str] = glob.glob("data/book_interactions_data/*/interactions.json", recursive=True)
    for interactions_file_path in interactions_files:
        book_path = Path(interactions_file_path)
        book_title = book_path.parent.name
        merged_interactions_path = book_path.parent / "interactions_merged.json"
            
        mapping = next((mapping for mapping in character_mapping if mapping[0] == book_title), None)
        if mapping is None:
            print(Fore.RED + f"No character mapping found for book: {book_title}" + Fore.RESET)
            continue

        print(Fore.GREEN + f"Matched '{book_title:40}' | '{mapping[1]:50}'" + Fore.RESET)
        merged_interactions = _merge_with_mapping(interactions_file_path, mapping[1])

        with open(merged_interactions_path, "w", encoding="utf-8") as output_file:
            json.dump(merged_interactions.model_dump(), output_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    generate_characters_summaries()
    merge_characters()
