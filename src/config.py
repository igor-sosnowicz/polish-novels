from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    max_books_per_epoch: int
    min_book_length: int

    llm_model: str
    max_chunk_size: int
    system_prompt: str

    ollama_options: dict[str, int | float]
    interactions_directory: Path
    graphs_directory: Path
    notebooks_directory: Path
    epoch_mapping: dict[str, str] = Field(default_factory=dict)

    # A character with relationship score below this threshold is considered an atagonist.
    atagonist_treshold: float
    # A character with relationship score above this threshold is considered a protagonist.
    protagonist_threshold: float


@lru_cache(maxsize=1)
def get_config() -> Config:
    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    return Config(**config_data)
