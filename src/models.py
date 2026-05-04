from enum import StrEnum
from pydantic import BaseModel, Field


class Sentiment(StrEnum):
    VERY_POSITIVE = "bardzo_pozytywna"
    POSITIVE = "pozytywna"
    NEUTRAL = "neutralna"
    NEGATIVE = "negatywna"
    VERY_NEGATIVE = "bardzo_negatywna"


class Interaction(BaseModel):
    character_1: str = Field(
        description="Imię lub określenie pierwszej postaci. MUSI to być człowiek (lub zwierzę). Nigdy nie używaj pojęć abstrakcyjnych ani 'N/A'. Zawsze w języku polskim."
    )
    character_2: str = Field(
        description="Imię lub określenie drugiej postaci. MUSI to być człowiek (lub zwierzę). Nigdy nie używaj pojęć abstrakcyjnych ani 'N/A'. Zawsze w języku polskim."
    )
    interaction_sentiment: Sentiment


class InteractionList(BaseModel):
    interactions: list[Interaction]

    def __add__(self, other: "InteractionList") -> "InteractionList":
        return InteractionList(interactions=self.interactions + other.interactions)

    def __sub__(self, other: "InteractionList") -> "InteractionList":
        return InteractionList(
            interactions=[
                interaction
                for interaction in self.interactions
                if interaction not in other.interactions
            ]
        )


class CharacterMapping(BaseModel):
    canonical_characters: list[str]
    canonical_non_human_personifiable_characters: list[str]
    alias_to_canonical: dict[str, str]
    drop_or_non_character: list[str]
