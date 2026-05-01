"""Module for text processing and inter-character relationship extraction."""

from enum import StrEnum, auto

from pydantic import BaseModel
from ollama import chat

from src.configuration import LLM_MODEL, MAX_CHUNK_SIZE


class Sentiment(StrEnum):
    VERY_POSITIVE = auto()
    POSITIVE = auto()
    NEUTRAL = auto()
    NEGATIVE = auto()
    VERY_NEGATIVE = auto()


class Interaction(BaseModel):
    character_1: str
    character_2: str
    interaction_sentiment: Sentiment


class InteractionList(BaseModel):
    interactions: list[Interaction]


def text_into_chunks(txt: str) -> list[str]:
    chunks = [chunk for chunk in txt.split("\n\n") if chunk]
    final_chunks = []
    for chunk in chunks:
        if len(chunk) < MAX_CHUNK_SIZE:
            final_chunks.append(chunk)
            continue

        # Split into smaller chunks each at most MAX_CHUNK_SIZE.
        sub_chunks = []
        current = ""
        for word in chunk.split():
            if len(current + " " + word) > MAX_CHUNK_SIZE:
                if current:
                    sub_chunks.append(current)
                    current = word
                else:
                    # If a single word is too long, add it anyway.
                    sub_chunks.append(word)
                    current = ""
            else:
                current += " " + word if current else word
        if current:
            sub_chunks.append(current)
        final_chunks.extend(sub_chunks)

    return final_chunks


def extract_relationships(text_chunk: str) -> list[Interaction]:

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Extract interactions between characters in a chunk of a novel provided by the user. Classify each interaction sentiment.",
            },
            {"role": "user", "content": f"Novel chunk:\n\n{text_chunk}"},
        ],
        format=InteractionList.model_json_schema(),
    )

    interactions = InteractionList.model_validate_json(response.message.content)
    for interaction in interactions:
        print(interaction)
        print("-" * 20)

    return []
