"""Generate a notebook with graph analysis results."""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import pandas as pd

from imports_setup import setup_project_imports

setup_project_imports()

from src.config import get_config
from src.graph_analyser import GraphAnalyser, GraphMetrics


REQUIRED_COLUMNS = {"character_1", "character_2", "interaction_sentiment"}


def _discover_graph_files(graphs_directory: Path) -> list[Path]:
    graph_files = sorted(graphs_directory.glob("*.csv"))
    if not graph_files:
        raise FileNotFoundError(f"No CSV graph files found in {graphs_directory}")
    return graph_files


def _validate_graph_frame(graph_file: Path) -> pd.DataFrame:
    df = pd.read_csv(graph_file)
    if df.empty:
        raise ValueError(f"{graph_file} is empty")

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{graph_file} is missing columns: {sorted(missing_columns)}")

    return df


def _load_epoch_mapping(config) -> dict[str, str]:
    if not getattr(config, "epoch_mapping", None):
        raise ValueError("config.yaml is missing epoch_mapping")
    return dict(config.epoch_mapping)


def _safe_name(name: str) -> str:
    return name.replace(" ", "_")


def _metrics_to_row(book_name: str, epoch: str, metrics: GraphMetrics, graph_file: Path) -> dict[str, object]:
    row = metrics.model_dump()
    row.update(
        {
            "book": book_name,
            "epoch": epoch,
            "graph_file": graph_file.name,
        }
    )
    return row


def _render_top_pairs(pairs: list[tuple[str, float]]) -> str:
    if not pairs:
        return "_No entries_"
    lines = ["| Character | Score |", "| --- | ---: |"]
    for character, score in pairs:
        lines.append(f"| {character} | {score:.2f} |")
    return "\n".join(lines)


def _markdown_list(items: list[str]) -> str:
    if not items:
        return "_No items_"
    return "\n".join(f"- {item}" for item in items)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"
    body_lines = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def _notebook_cell(cell_type: str, language: str, source: list[str]) -> dict[str, object]:
    cell: dict[str, object] = {
        "cell_type": cell_type,
        "metadata": {
            "id": uuid4().hex,
            "language": language,
        },
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with path.open("w", encoding="utf-8") as notebook_file:
        json.dump(notebook, notebook_file, ensure_ascii=False, indent=2)


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Path]], dict[str, Path], Path]:
    config = get_config()
    graphs_directory = Path(config.graphs_directory)
    notebooks_directory = Path(config.notebooks_directory)
    plots_directory = notebooks_directory / "plots"
    plots_directory.mkdir(parents=True, exist_ok=True)
    notebooks_directory.mkdir(parents=True, exist_ok=True)

    if not graphs_directory.exists():
        raise FileNotFoundError(f"Graphs directory does not exist: {graphs_directory}")

    epoch_mapping = _load_epoch_mapping(config)
    graph_files = _discover_graph_files(graphs_directory)
    graph_analyser = GraphAnalyser()

    rows: list[dict[str, object]] = []
    epoch_to_metrics: dict[str, list[GraphMetrics]] = defaultdict(list)
    epoch_to_books: dict[str, list[str]] = defaultdict(list)
    book_to_plots: dict[str, dict[str, Path]] = defaultdict(dict)

    for graph_file in graph_files:
        frame = _validate_graph_frame(graph_file)
        book_name = graph_file.stem
        if book_name not in epoch_mapping:
            raise KeyError(f"No epoch mapping found for {book_name}")

        epoch = epoch_mapping[book_name]
        metrics = graph_analyser.calculate_metrics(frame)
        protagonists = graph_analyser.find_protagonists(frame, min(5, max(1, len(frame))))
        antagonists = graph_analyser.find_antagonists(frame, min(5, max(1, len(frame))))

        rows.append(_metrics_to_row(book_name, epoch, metrics, graph_file))
        rows[-1]["protagonists"] = _render_top_pairs(protagonists)
        rows[-1]["antagonists"] = _render_top_pairs(antagonists)
        epoch_to_metrics[epoch].append(metrics)
        epoch_to_books[epoch].append(book_name)

        book_to_plots[book_name]["degree_histogram"] = graph_analyser.create_degree_histogram(frame, filename=f"{_safe_name(book_name)}_degree_histogram.png")
        book_to_plots[book_name]["degree"] = graph_analyser.visualise_graph(frame, parameter="degree", filename=f"{_safe_name(book_name)}_degree_graph.png")
        book_to_plots[book_name]["page_rank"] = graph_analyser.visualise_graph(frame, parameter="page_rank", filename=f"{_safe_name(book_name)}_page_rank_graph.png")
        book_to_plots[book_name]["betweeness"] = graph_analyser.visualise_graph(frame, parameter="betweeness", filename=f"{_safe_name(book_name)}_betweeness_graph.png")
        book_to_plots[book_name]["closeness"] = graph_analyser.visualise_graph(frame, parameter="closeness", filename=f"{_safe_name(book_name)}_closeness_graph.png")
        book_to_plots[book_name]["absolute_relationshipscore"] = graph_analyser.visualise_graph(frame, parameter="absolute_relationshipscore", filename=f"{_safe_name(book_name)}_absolute_relationshipscore_graph.png")

    summary_df = pd.DataFrame(rows).sort_values(["epoch", "book"]).reset_index(drop=True)
    summary_path = notebooks_directory / "graph_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    epoch_summary_df = (
        summary_df.groupby("epoch")
        .agg(
            books=("book", "count"),
            diameter_mean=("diameter", "mean"),
            max_degree_mean=("max_degree", "mean"),
            clustering_mean=("clustering_coefficient", "mean"),
            betweenness_mean=("betweeness", "mean"),
            closeness_mean=("closeness", "mean"),
            page_rank_mean=("page_rank", "mean"),
            relationship_score_mean=("average_relationship_score", "mean"),
            protagonist_count_mean=("protagonist_count", "mean"),
            antagonist_count_mean=("antagonist_count", "mean"),
        )
        .reset_index()
        .sort_values("epoch")
    )
    epoch_summary_path = notebooks_directory / "epoch_metrics_summary.csv"
    epoch_summary_df.to_csv(epoch_summary_path, index=False)

    epoch_representative_book: dict[str, str] = {}
    for epoch, group_df in summary_df.groupby("epoch"):
        epoch_representative_book[epoch] = group_df.sort_values(["max_degree", "page_rank"], ascending=False).iloc[0]["book"]

    pairwise_comparison_plots: dict[str, Path] = {}
    for epoch_a, epoch_b in combinations(sorted(epoch_to_metrics), 2):
        pair_key = f"{epoch_a}__vs__{epoch_b}"
        pairwise_comparison_plots[pair_key] = graph_analyser.compare_graph_groups(epoch_a, epoch_to_metrics[epoch_a], epoch_b, epoch_to_metrics[epoch_b])

    notebook_path = notebooks_directory / "graph_analysis.ipynb"
    cells: list[dict[str, object]] = []

    cells.append(_notebook_cell("markdown", "markdown", ["# Graph analysis report\n"]))
    cells.append(_notebook_cell("markdown", "markdown", ["This notebook is generated from the graph CSV files in `data/graphs/`.\n", "\n", "It includes per-book metrics, per-epoch aggregation, pairwise comparisons, and representative plots generated by `GraphAnalyser`.\n"]))
    cells.append(_notebook_cell("code", "python", ["from pathlib import Path\n", "import pandas as pd\n", "from IPython.display import display\n", "\n", "summary = pd.read_csv(Path('graph_metrics_summary.csv'))\n", "epoch_summary = pd.read_csv(Path('epoch_metrics_summary.csv'))\n", "display(summary)\n", "display(epoch_summary)\n"]))

    for epoch, books in sorted(epoch_to_books.items()):
        epoch_books_df = summary_df[summary_df["epoch"] == epoch].copy()
        representative_book = epoch_representative_book[epoch]
        representative_plots = book_to_plots[representative_book]

        cells.append(_notebook_cell("markdown", "markdown", [f"## Epoch: {epoch}\n"]))
        representative_plot_lines = [
            f"{label}: `plots/{path.name}`" for label, path in representative_plots.items()
        ]
        cells.append(_notebook_cell("markdown", "markdown", [f"**Representative book:** {representative_book}\n", "\n", "Books:\n", _markdown_list(books), "\n", "**Representative plots**\n", "\n", _markdown_list(representative_plot_lines), "\n", f"![{epoch} representative graph](plots/{representative_plots['page_rank'].name})\n"]))
        cells.append(_notebook_cell("markdown", "markdown", ["### Per-book metrics\n", "\n", _markdown_table(["book", "diameter", "max_degree", "page_rank", "protagonist_count", "antagonist_count"], epoch_books_df[["book", "diameter", "max_degree", "page_rank", "protagonist_count", "antagonist_count"]].astype(str).values.tolist()), "\n"]))

    cells.append(_notebook_cell("markdown", "markdown", ["## Pairwise epoch comparisons\n"]))
    for comparison_name, plot_path in sorted(pairwise_comparison_plots.items()):
        epoch_a, epoch_b = comparison_name.split("__vs__", 1)
        cells.append(_notebook_cell("markdown", "markdown", [f"### {epoch_a} vs {epoch_b}\n", "\n", f"![{comparison_name}]({plot_path.as_posix()})\n"]))

    _write_notebook(notebook_path, cells)

    return summary_df, epoch_summary_df, book_to_plots, pairwise_comparison_plots, notebook_path


def run_sanity_checks() -> dict[str, object]:
    config = get_config()
    graphs_directory = Path(config.graphs_directory)
    notebooks_directory = Path(config.notebooks_directory)
    notebooks_directory.mkdir(parents=True, exist_ok=True)

    if not graphs_directory.exists():
        raise FileNotFoundError(f"Graphs directory does not exist: {graphs_directory}")

    graph_files = _discover_graph_files(graphs_directory)
    graph_analyser = GraphAnalyser()

    sample_file = graph_files[0]
    sample_frame = _validate_graph_frame(sample_file)

    return {
        "graph_files_count": len(graph_files),
        "sample_file": sample_file.name,
        "sample_rows": len(sample_frame),
        "sample_columns": list(sample_frame.columns),
        "graph_analyser_ready": isinstance(graph_analyser, GraphAnalyser),
    }


def main() -> None:
    summary_df, epoch_summary_df, book_to_plots, comparison_plots, notebook_path = run_analysis()

    print("Analysis completed")
    print(f"Books analysed: {len(summary_df)}")
    print(f"Epochs analysed: {len(epoch_summary_df)}")
    print(f"Notebook written to: {notebook_path}")
    print(f"Generated book plot groups: {len(book_to_plots)}")
    print(f"Generated comparison plot groups: {len(comparison_plots)}")


if __name__ == "__main__":
    main()