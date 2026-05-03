"""Script for computing network features from all graphs and saving to CSV."""


import pandas as pd
from loguru import logger
from tqdm import tqdm

from novels_analysis.config.configuration import FEATURES_CSV, GRAPHS_DIR
from novels_analysis.graph.graph_analysis import extract_all_features
from novels_analysis.graph.graph_builder import load_graph


def main() -> None:
    epoch_dirs = sorted(p for p in GRAPHS_DIR.iterdir() if p.is_dir())

    if not epoch_dirs:
        logger.error(
            f"No epoch directories found in {GRAPHS_DIR}. "
            "Run build_graphs.py first."
        )
        return

    rows: list[dict] = []

    for epoch_dir in tqdm(epoch_dirs, desc="Epochs"):
        epoch = epoch_dir.name
        graph_files = sorted(epoch_dir.glob("*.graphml"))

        for graph_path in tqdm(graph_files, desc=epoch, leave=False):
            book = graph_path.stem
            try:
                graph = load_graph(graph_path)
            except Exception as exc:
                logger.warning(f"Could not load {graph_path}: {exc}")
                continue

            features = extract_all_features(graph, book=book, epoch=epoch)
            rows.append(features)
            logger.debug(f"Extracted features for {epoch}/{book}")

    if not rows:
        logger.error("No features extracted. Check your graphs directory.")
        return

    df = pd.DataFrame(rows)

    # Put identifier columns first
    id_cols = ["book", "epoch"]
    feature_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + feature_cols]

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False, encoding="utf-8")

    logger.success(
        f"Saved features for {len(df)} books across "
        f"{df['epoch'].nunique()} epochs → {FEATURES_CSV}"
    )
    logger.info(f"Feature columns: {feature_cols}")
    logger.info(f"\n{df.groupby('epoch').size().rename('n_books').to_string()}")


if __name__ == "__main__":
    main()