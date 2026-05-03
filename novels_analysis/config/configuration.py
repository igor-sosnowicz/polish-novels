"""Module with the configuration of the system."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Book fetching
MIN_BOOK_LENGTH = 500_000  # in characters
MAX_BOOKS_PER_EPOCH = 5

# Text processing
MAX_CHUNK_SIZE = 1000   # in characters
CHUNK_OVERLAP = 200

# LLM
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Paths
DATA_DIR = Path("data")
BOOKS_DIR = DATA_DIR / "books"
RELATIONSHIPS_DIR = DATA_DIR / "relationships"
GRAPHS_DIR = DATA_DIR / "graphs"
FEATURES_CSV = DATA_DIR / "features.csv"