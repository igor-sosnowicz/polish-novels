from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    max_books_per_epoch: int
    min_book_length: int

    llm_model: str
    max_chunk_size: int
    system_prompt: str

    ollama_options: dict[str, int | float]
    interactions_directory: Path


@lru_cache(maxsize=1)
def get_config() -> Config:
    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    return Config(**config_data)
