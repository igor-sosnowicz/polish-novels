"""Module for building NetworkX graphs from extracted character interactions."""

from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger

from novels_analysis.process.text_processing import SENTIMENT_SCORE


def interactions_to_graph(
    interactions: list[dict[str, Any]],
    title: str = "",
    epoch: str = "",
) -> nx.Graph:
    """
    Build a weighted undirected graph from a list of interaction dicts.

    Node attributes:
        - mention_count: how many interactions involve this character

    Edge attributes:
        - weight: number of interactions between the pair
        - sentiment_score: mean sentiment score across all interactions
        - positive_count / negative_count: breakdown by polarity
    """
    graph = nx.Graph(title=title, epoch=epoch)

    edge_data: dict[tuple[str, str], dict[str, Any]] = {}

    for item in interactions:
        c1: str = item["character_1"]
        c2: str = item["character_2"]

        # Skip self-loops or empty names
        if not c1 or not c2 or c1 == c2:
            continue

        # Canonical ordering so (A, B) and (B, A) map to the same key
        key = (min(c1, c2), max(c1, c2))

        sentiment_str: str = item.get("interaction_sentiment", "neutral")
        score = SENTIMENT_SCORE.get(sentiment_str, 0.0)

        if key not in edge_data:
            edge_data[key] = {"scores": [], "positive": 0, "negative": 0}

        edge_data[key]["scores"].append(score)
        if score > 0:
            edge_data[key]["positive"] += 1
        elif score < 0:
            edge_data[key]["negative"] += 1

        for char in (c1, c2):
            if char not in graph:
                graph.add_node(char, mention_count=0)
            graph.nodes[char]["mention_count"] += 1

    for (c1, c2), data in edge_data.items():
        scores = data["scores"]
        mean_sentiment = sum(scores) / len(scores)
        graph.add_edge(
            c1,
            c2,
            weight=len(scores),
            sentiment_score=round(mean_sentiment, 4),
            positive_count=data["positive"],
            negative_count=data["negative"],
        )

    logger.debug(
        f"Graph '{title}': {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges."
    )
    return graph


def save_graph(graph: nx.Graph, path: Path) -> None:
    """Save graph to GraphML format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)
    logger.debug(f"Graph saved -> {path}")


def load_graph(path: Path) -> nx.Graph:
    """Load graph from GraphML format."""
    return nx.read_graphml(path)