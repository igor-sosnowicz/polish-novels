"""Script for building NetworkX graphs from extracted relationship JSON files."""

import json

from loguru import logger
from tqdm import tqdm

from novels_analysis.config.configuration import GRAPHS_DIR, RELATIONSHIPS_DIR
from novels_analysis.graph.graph_builder import interactions_to_graph, save_graph


def main() -> None:
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    epoch_dirs = sorted(p for p in RELATIONSHIPS_DIR.iterdir() if p.is_dir())

    if not epoch_dirs:
        logger.error(
            f"No epoch directories found in {RELATIONSHIPS_DIR}. "
            "Run extract_relationships.py first."
        )
        return

    for epoch_dir in tqdm(epoch_dirs, desc="Epochs"):
        epoch = epoch_dir.name
        out_dir = GRAPHS_DIR / epoch
        out_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(epoch_dir.glob("*.json"))

        for json_path in tqdm(json_files, desc=f"{epoch}", leave=False):
            out_path = out_dir / json_path.with_suffix(".graphml").name

            if out_path.exists():
                logger.debug(f"Skipping {json_path.name} (graph already built).")
                continue

            interactions = json.loads(json_path.read_text(encoding="utf-8"))

            if not interactions:
                logger.warning(f"No interactions in {json_path.name}, skipping.")
                continue

            title = json_path.stem
            graph = interactions_to_graph(interactions, title=title, epoch=epoch)

            if graph.number_of_nodes() < 2:
                logger.warning(f"Graph for '{title}' has fewer than 2 nodes, skipping.")
                continue

            save_graph(graph, out_path)
            logger.success(f"Saved graph → {out_path}")


if __name__ == "__main__":
    main()