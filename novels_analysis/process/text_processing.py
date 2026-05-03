"""Module for text processing and inter-character relationship extraction."""

from enum import StrEnum
from typing import Any

from google import genai
from loguru import logger
from pydantic import BaseModel

from novels_analysis.config.configuration import CHUNK_OVERLAP, GEMINI_API_KEY, GEMINI_MODEL, MAX_CHUNK_SIZE

_client = genai.Client(api_key=GEMINI_API_KEY)


class Sentiment(StrEnum):
    VERY_POSITIVE: str = "very_positive"
    POSITIVE: str = "positive"
    NEUTRAL: str = "neutral"
    NEGATIVE: str = "negative"
    VERY_NEGATIVE: str = "very_negative"


SENTIMENT_SCORE: dict[Sentiment, float] = {
    Sentiment.VERY_POSITIVE: 1.0,
    Sentiment.POSITIVE: 0.5,
    Sentiment.NEUTRAL: 0.0,
    Sentiment.NEGATIVE: -0.5,
    Sentiment.VERY_NEGATIVE: -1.0,
}


class Interaction(BaseModel):
    character_1: str
    character_2: str
    interaction_sentiment: Sentiment


class InteractionList(BaseModel):
    interactions: list[Interaction]


def text_into_chunks(txt: str) -> list[str]:
    """Split text into overlapping chunks of at most MAX_CHUNK_SIZE characters."""
    words = txt.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word) + 1
        if current_len + word_len > MAX_CHUNK_SIZE and current_words:
            chunk = " ".join(current_words)
            chunks.append(chunk)
            # Keep last CHUNK_OVERLAP characters worth of words for overlap
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(current_words):
                if overlap_len + len(w) + 1 > CHUNK_OVERLAP:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
            current_words = overlap_words
            current_len = overlap_len

        current_words.append(word)
        current_len += word_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks

# Gemini says it's the best version for model we are using ;)
SYSTEM_PROMPT: str = (
    "You are an expert literary analyst specializing in Polish narrative structures. "
    "Your task is to extract character interactions from the provided text fragments. "
    "\n\n### Task Requirements:\n"
    "1. **Identify Interactions**: Detect direct interactions between literary characters (e.g., dialogues, physical actions, direct meetings). "
    "Ignore mentions of characters who do not actively engage with one another in the scene.\n"
    "2. **Entity Normalization**: Extract character names and normalize them to the Polish nominative case (mianownik), "
    "regardless of how they appear in the text (e.g., 'Kmicicem' -> 'Kmicic').\n"
    "3. **Sentiment Analysis**: Determine the sentiment of each specific interaction (e.g., 'positive', 'negative', or 'neutral').\n"
    "4. **Output Format**: Return the data strictly as a JSON object. Do not include any conversational filler or markdown outside the JSON block.\n\n"
    "### JSON Schema:\n"
    "{\n"
    '  "interactions": [\n'
    "    {\n"
    '      "source": "Character A (Nominative)",\n'
    '      "target": "Character B (Nominative)",\n'
    '      "sentiment": "positive/negative/neutral",\n'
    '      "description": "Short justification of the interaction"\n'
    "    }\n"
    "  ]\n"
    "}"
)

def extract_relationships(text_chunk: str) -> list[Interaction]:
    """Extract character interactions from a single text chunk using Gemini."""
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Fragment of book:\n\n{text_chunk}",
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=InteractionList,
                temperature=0.0,
            ),
        )
        result = InteractionList.model_validate_json(response.text)
        return result.interactions
    except Exception as exc:
        logger.warning(f"Failed to extract relationships from chunk: {exc}")
        return []


def _normalize_name(name: str) -> str:
    """Basic normalization: strip whitespace, title-case."""
    return name.strip().title()


def process_book(txt: str) -> list[dict[str, Any]]:
    """
    Process an entire book text and return a list of interaction dicts.
    Each dict is JSON-serializable and ready to be saved.
    """
    chunks = text_into_chunks(txt)
    all_interactions: list[dict[str, Any]] = []

    for chunk in chunks:
        interactions = extract_relationships(chunk)
        for interaction in interactions:
            all_interactions.append({
                "character_1": _normalize_name(interaction.character_1),
                "character_2": _normalize_name(interaction.character_2),
                "interaction_sentiment": interaction.interaction_sentiment,
            })

    return all_interactions