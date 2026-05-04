"""The script for converting processed interactions to graphs."""

import tarfile
from pathlib import Path

from imports_setup import setup_project_imports

setup_project_imports()

from src.graph_converter import GraphConverter
from src.config import get_config


def main() -> None:
    config = get_config()

    # Decompress `data_interactions.tar.gz` into `./data` directory.
    tar_file = Path("data_interactions.tar.gz")
    if tar_file.exists():
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=".")

    # Interactions to adjacency list convertion.
    graph_converter = GraphConverter()
    output_directory = Path("./data/graphs/")

    for directory in config.interactions_directory.iterdir():
        filepath = directory / "interactions.json"
        output_file = output_directory / f"{directory.name}.csv"
        df = graph_converter(filepath)

        # Save to CSV.
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
