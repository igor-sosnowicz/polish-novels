"""Module for computing complex network metrics and ML features from graphs."""

from __future__ import annotations

import warnings
from typing import Any

import networkx as nx
import numpy as np

# python-louvain exposes its API under `community`
try:
    import community as community_louvain
    _LOUVAIN_AVAILABLE = True
except ImportError:
    _LOUVAIN_AVAILABLE = False


def basic_stats(graph: nx.Graph) -> dict[str, Any]:
    """Node/edge counts, density, diameter, average shortest path."""
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    density = nx.density(graph)

    if n_nodes < 2:
        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "diameter": 0,
            "avg_shortest_path": 0.0,
            "n_components": 0,
        }

    components = list(nx.connected_components(graph))
    n_components = len(components)
    largest_cc = graph.subgraph(max(components, key=len)).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            diameter = nx.diameter(largest_cc)
            avg_path = nx.average_shortest_path_length(largest_cc)
        except Exception:
            diameter = 0
            avg_path = 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": round(density, 6),
        "diameter": diameter,
        "avg_shortest_path": round(avg_path, 4),
        "n_components": n_components,
    }


def degree_stats(graph: nx.Graph) -> dict[str, Any]:
    """
    Degree distribution stats.
    High skewness + low mean -> scale-free-like (hub characters dominate).
    """
    if graph.number_of_nodes() == 0:
        return {
            "avg_degree": 0.0,
            "std_degree": 0.0,
            "max_degree": 0,
            "degree_skewness": 0.0,
        }

    degrees = np.array([d for _, d in graph.degree()])
    avg = float(np.mean(degrees))
    std = float(np.std(degrees))
    max_deg = int(np.max(degrees))

    skewness = float(np.mean(((degrees - avg) / std) ** 3)) if std > 0 else 0.0

    return {
        "avg_degree": round(avg, 4),
        "std_degree": round(std, 4),
        "max_degree": max_deg,
        "degree_skewness": round(skewness, 4),
    }


def centrality_features(graph: nx.Graph) -> dict[str, Any]:
    """
    Top-character centrality scores (protagonists by network position).
    Returns aggregate stats over all nodes rather than per-node values
    so the output is a fixed-width feature vector.
    """
    if graph.number_of_nodes() < 2:
        return {
            "avg_betweenness": 0.0,
            "max_betweenness": 0.0,
            "avg_closeness": 0.0,
            "max_closeness": 0.0,
            "avg_pagerank": 0.0,
            "max_pagerank": 0.0,
        }

    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    closeness = nx.closeness_centrality(graph)
    pagerank = nx.pagerank(graph, weight="weight")

    def _stats(mapping: dict[str, float]) -> tuple[float, float]:
        vals = list(mapping.values())
        return round(float(np.mean(vals)), 6), round(float(np.max(vals)), 6)

    b_avg, b_max = _stats(betweenness)
    c_avg, c_max = _stats(closeness)
    p_avg, p_max = _stats(pagerank)

    return {
        "avg_betweenness": b_avg,
        "max_betweenness": b_max,
        "avg_closeness": c_avg,
        "max_closeness": c_max,
        "avg_pagerank": p_avg,
        "max_pagerank": p_max,
    }


def clustering_features(graph: nx.Graph) -> dict[str, Any]:
    """Clustering coefficient and community detection (Louvain)."""
    if graph.number_of_nodes() < 2:
        return {
            "avg_clustering": 0.0,
            "transitivity": 0.0,
            "n_communities": 0,
            "modularity": 0.0,
        }

    avg_clustering = nx.average_clustering(graph, weight="weight")
    transitivity = nx.transitivity(graph)

    n_communities = 0
    modularity = 0.0
    if _LOUVAIN_AVAILABLE and graph.number_of_edges() > 0:
        try:
            partition = community_louvain.best_partition(graph, weight="weight")
            n_communities = len(set(partition.values()))
            modularity = community_louvain.modularity(partition, graph, weight="weight")
        except Exception:
            pass

    return {
        "avg_clustering": round(avg_clustering, 6),
        "transitivity": round(transitivity, 6),
        "n_communities": n_communities,
        "modularity": round(modularity, 6),
    }


def sentiment_features(graph: nx.Graph) -> dict[str, Any]:
    """
    Sentiment-based features derived from edge attributes.
    Key for H3: ratio of negative edges and antagonist node count.
    An 'antagonist' node has more negative-weight edges than positive ones.
    """
    edges = list(graph.edges(data=True))

    if not edges:
        return {
            "avg_sentiment": 0.0,
            "negative_edge_ratio": 0.0,
            "positive_edge_ratio": 0.0,
            "n_antagonist_nodes": 0,
            "antagonist_ratio": 0.0,
        }

    scores = [d.get("sentiment_score", 0.0) for _, _, d in edges]
    avg_sentiment = float(np.mean(scores))

    negative_edges = sum(1 for s in scores if s < 0)
    positive_edges = sum(1 for s in scores if s > 0)
    total = len(scores)

    antagonist_count = 0
    for node in graph.nodes():
        node_edges = graph.edges(node, data=True)
        pos = sum(1 for _, _, d in node_edges if d.get("sentiment_score", 0.0) > 0)
        neg = sum(1 for _, _, d in node_edges if d.get("sentiment_score", 0.0) < 0)
        if neg > pos:
            antagonist_count += 1

    n_nodes = graph.number_of_nodes()

    return {
        "avg_sentiment": round(avg_sentiment, 4),
        "negative_edge_ratio": round(negative_edges / total, 4),
        "positive_edge_ratio": round(positive_edges / total, 4),
        "n_antagonist_nodes": antagonist_count,
        "antagonist_ratio": round(antagonist_count / n_nodes, 4) if n_nodes else 0.0,
    }


def extract_all_features(
    graph: nx.Graph,
    book: str = "",
    epoch: str = "",
) -> dict[str, Any]:
    """
    Compute all feature groups and return a single flat dict.
    Each dict represents one row in the final features DataFrame.
    """
    features: dict[str, Any] = {"book": book, "epoch": epoch}
    features.update(basic_stats(graph))
    features.update(degree_stats(graph))
    features.update(centrality_features(graph))
    features.update(clustering_features(graph))
    features.update(sentiment_features(graph))
    return features