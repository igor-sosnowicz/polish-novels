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