"""Module for loading interactions to a graph."""

import json
from pathlib import Path

import pandas as pd


class GraphConverter:
    def __call__(self, book_file: Path) -> pd.DataFrame:
        with book_file.open("rt") as file:
            interactions = json.load(file)["interactions"]

        df = pd.DataFrame.from_records(interactions)
        # Drop all N/A characters.
        df = df.drop(
            df[(df["character_1"] == "N/A") | (df["character_2"] == "N/A")].index
        )

        return df
