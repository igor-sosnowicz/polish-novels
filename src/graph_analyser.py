"""Module for analysing one graph or comparing several graphs."""

from pathlib import Path
from typing import Literal, Tuple

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel

from src.config import get_config


class GraphMetrics(BaseModel):
    diameter: float | None
    max_degree: int
    clustering_coefficient: float
    betweeness: float
    closeness: float
    page_rank: float
    average_relationship_score: float
    protagonist_count: int
    antagonist_count: int


class GraphAnalyser:
    SENTIMENT_WEIGHT = {
        "bardzo_negatywna": -2,
        "negatywna": -1,
        "neutralna": 0,
        "pozytywna": 1,
        "bardzo_pozytywna": 2,
    }

    def _build_graph(self, adjacency_list: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        for _, row in adjacency_list.iterrows():
            a = str(row["character_1"]).strip()
            b = str(row["character_2"]).strip()
            weight = self.SENTIMENT_WEIGHT.get(
                str(row["interaction_sentiment"]).strip(), 0
            )
            if G.has_edge(a, b):
                G[a][b]["weight"] += weight
                G[a][b]["count"] += 1
            else:
                G.add_edge(a, b, weight=weight, count=1)
        return G

    def _pagerank_scores(self, graph: nx.Graph) -> dict[str, float]:
        try:
            return nx.pagerank(graph, weight="count", max_iter=1000, tol=1e-08)
        except Exception:
            return nx.pagerank(graph, weight=None, max_iter=1000, tol=1e-08)

    def _node_metric_values(
        self,
        graph: nx.Graph,
        parameter: Literal[
            "page_rank",
            "betweeness",
            "closeness",
            "degree",
            "absolute_relationshipscore",
            "absolute_relationship_score",
        ]
        | None,
    ) -> dict[str, float]:
        if parameter == "page_rank":
            return self._pagerank_scores(graph)
        if parameter == "betweeness":
            return nx.betweenness_centrality(graph)
        if parameter == "closeness":
            return nx.closeness_centrality(graph)
        if parameter in {"absolute_relationshipscore", "absolute_relationship_score"}:
            return {
                node: sum(
                    abs(d.get("weight", 0)) for _, _, d in graph.edges(node, data=True)
                )
                for node in graph.nodes()
            }
        return dict(graph.degree())

    def _node_sizes(
        self, metric_values: dict[str, float], graph: nx.Graph
    ) -> list[float]:
        values = [
            max(0.0, float(metric_values.get(node, 0.0))) for node in graph.nodes()
        ]
        if not values:
            return []

        max_value = max(values)
        min_value = min(values)

        # If the selected metric is nearly flat, fall back to weighted node strength
        # so the plot still reveals the network structure.
        if max_value <= 0 or abs(max_value - min_value) < 1e-12:
            values = [
                sum(
                    abs(d.get("weight", 0)) + d.get("count", 0)
                    for _, _, d in graph.edges(node, data=True)
                )
                for node in graph.nodes()
            ]
            max_value = max(values) if values else 0.0
            min_value = min(values) if values else 0.0

        if max_value <= 0 or abs(max_value - min_value) < 1e-12:
            return [50.0 for _ in values]

        spread = max_value - min_value
        return [
            10.0 + 290.0 * (((value - min_value) / spread) ** 1.0) for value in values
        ]

    def _graph_layout(self, graph: nx.Graph) -> dict[str, tuple[float, float]]:
        if len(graph) == 0:
            return {}
        import random

        n = len(graph)
        k = 1.6 / max(n**0.5, 1.0)

        try:
            pos = nx.spring_layout(
                graph,
                seed=42,
                weight="count",
                k=k,
                iterations=500,
                scale=1.0,
            )
        except Exception:
            try:
                pos = nx.kamada_kawai_layout(graph, weight="count")
            except Exception:
                pos = {node: (0.0, 0.0) for node in graph.nodes()}

        jitter = 0.002
        normalized_pos: dict[str, tuple[float, float]] = {}
        for node, raw_xy in pos.items():
            x = float(raw_xy[0])
            y = float(raw_xy[1])
            normalized_pos[str(node)] = (
                x + random.uniform(-jitter, jitter),
                y + random.uniform(-jitter, jitter),
            )

        return normalized_pos

    def calculate_metrics(self, adjacency_list: pd.DataFrame) -> GraphMetrics:
        config = get_config()
        # tolerant access for possible misspelled keys
        protagonist_threshold = getattr(config, "protagonist_threshold", 0.5)
        antagonist_threshold = getattr(config, "atagonist_treshold", -0.5)

        G = self._build_graph(adjacency_list)

        if len(G) == 0:
            raise ValueError("Empty graph has been created.")

        # Work on the largest connected component for metrics that require connectivity
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            Gc = G.subgraph(largest_cc).copy()
        else:
            Gc = G

        try:
            diameter = float(nx.diameter(Gc))
        except Exception:
            diameter = None

        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 0

        clustering = nx.average_clustering(G, weight="count") if len(G) > 0 else 0.0

        betweenness = nx.betweenness_centrality(G)
        avg_betweenness = (
            float(sum(betweenness.values()) / len(betweenness)) if betweenness else 0.0
        )

        closeness = nx.closeness_centrality(G)
        avg_closeness = (
            float(sum(closeness.values()) / len(closeness)) if closeness else 0.0
        )

        pagerank = self._pagerank_scores(G)
        avg_pagerank = (
            float(sum(pagerank.values()) / len(pagerank)) if pagerank else 0.0
        )

        # average relationship score per edge
        edge_weights = [d.get("weight", 0) for u, v, d in G.edges(data=True)]
        avg_rel = float(sum(edge_weights) / len(edge_weights)) if edge_weights else 0.0

        # node scores: sum of incident edge weights
        node_scores = {
            n: sum(d.get("weight", 0) for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }

        protagonist_count = sum(
            1 for s in node_scores.values() if s >= protagonist_threshold
        )
        antagonist_count = sum(
            1 for s in node_scores.values() if s <= antagonist_threshold
        )

        return GraphMetrics(
            diameter=diameter,
            max_degree=max_degree,
            clustering_coefficient=float(clustering),
            betweeness=avg_betweenness,
            closeness=avg_closeness,
            page_rank=avg_pagerank,
            average_relationship_score=avg_rel,
            protagonist_count=protagonist_count,
            antagonist_count=antagonist_count,
        )

    def create_degree_histogram(
        self, adjacency_list: pd.DataFrame, *, filename: str | None = None
    ) -> Path:
        config = get_config()
        out_dir = Path(config.notebooks_directory) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        G = self._build_graph(adjacency_list)
        degrees = [d for _, d in G.degree()]

        plt.figure(figsize=(6, 4))
        sns.histplot(degrees, bins=20, kde=False)
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.tight_layout()

        if filename is None:
            filename = "degree_histogram.png"
        out_path = out_dir / filename
        plt.savefig(out_path)
        plt.close()
        return out_path

    def visualise_graph(
        self,
        adjacency_list: pd.DataFrame,
        parameter: Literal[
            "page_rank",
            "betweeness",
            "closeness",
            "degree",
            "absolute_relationshipscore",
            "absolute_relationship_score",
        ]
        | None = None,
        *,
        filename: str | None = None,
    ) -> Path:
        config = get_config()
        out_dir = Path(config.notebooks_directory) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        G = self._build_graph(adjacency_list)
        metric = self._node_metric_values(G, parameter)
        sizes = self._node_sizes(metric, G)
        widths = [
            0.3 + 0.8 * (d.get("count", 1) ** 0.5) for _, _, d in G.edges(data=True)
        ]

        # Cap sizes to avoid excessive overlap and ensure visibility for small nodes
        sizes = [max(6.0, min(float(s), 200.0)) for s in sizes]

        # compute layout (may be expensive for large graphs)
        pos = self._graph_layout(G)
        plt.figure(figsize=(12, 9))

        # Draw edges first (beneath nodes) with reduced alpha and thin lines
        edge_artist = nx.draw_networkx_edges(
            G, pos, alpha=0.2, width=widths, edge_color="#666666"
        )
        try:
            edge_artist.set_zorder(1)
        except Exception:
            pass

        # Draw nodes on top, semi-transparent, with a thin border to separate overlapping points
        node_artist = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=sizes,
            alpha=0.75,
            node_color="#1f77b4",
            linewidths=0.25,
            edgecolors="#222222",
        )
        try:
            node_artist.set_zorder(2)
        except Exception:
            pass
        plt.axis("off")

        if filename is None:
            filename = f"graph_vis_{parameter or 'degree'}.png"
        out_path = out_dir / filename
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def export_pyvis_graph(
        self,
        adjacency_list: pd.DataFrame,
        parameter: Literal[
            "page_rank",
            "betweeness",
            "closeness",
            "degree",
            "absolute_relationshipscore",
            "absolute_relationship_score",
        ]
        | None = "degree",
        *,
        filename: str | None = None,
        largest_component_only: bool = False,
    ) -> Path:
        from pyvis.network import Network

        config = get_config()
        out_dir = Path(config.notebooks_directory) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        G = self._build_graph(adjacency_list)
        if largest_component_only and len(G) > 0 and not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        metric = self._node_metric_values(G, parameter)
        metric_values = [float(metric.get(node, 0.0)) for node in G.nodes()]
        max_metric = max(metric_values) if metric_values else 0.0

        net = Network(height="850px", width="100%", notebook=False)
        net.barnes_hut()

        for node in G.nodes():
            value = max(0.0, float(metric.get(node, 0.0)))
            if max_metric > 0:
                norm = value / max_metric
            else:
                norm = 0.0

            size = 6.0 + 20.0 * norm
            red = int(255 - 200 * norm)
            color = f"rgb({red},120,150)"
            degree = G.degree(node)

            net.add_node(
                str(node),
                label=str(node),
                title=f"{node}<br/>Degree: {degree}<br/>{parameter or 'degree'}: {value:.4f}",
                size=size,
                color=color,
            )

        for u, v, data in G.edges(data=True):
            count = int(data.get("count", 1))
            width = max(1.0, 0.8 + count**0.5)
            net.add_edge(str(u), str(v), value=width, title=f"count={count}")

        if filename is None:
            filename = f"graph_vis_{parameter or 'degree'}.html"
        out_path = out_dir / filename
        net.write_html(str(out_path), notebook=False)
        return out_path

    def find_protagonists(
        self, adjacency_list: pd.DataFrame, k: int
    ) -> list[Tuple[str, float]]:
        G = self._build_graph(adjacency_list)
        node_scores = {
            n: sum(d.get("weight", 0) for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }
        top = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return top

    def find_antagonists(
        self, adjacency_list: pd.DataFrame, k: int
    ) -> list[tuple[str, float]]:
        G = self._build_graph(adjacency_list)
        node_scores = {
            n: sum(d.get("weight", 0) for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }
        return sorted(node_scores.items(), key=lambda x: x[1])[:k]

    @staticmethod
    def compare_graph_groups(
        group_A_name: str,
        group_A: list[GraphMetrics],
        group_B_name: str,
        group_B: list[GraphMetrics],
    ) -> Path:
        # Compare two groups of graphs. Calculate averages, min, max, stdev and plots on subplots comparing one group to the other.
        if not group_A or not group_B:
            raise ValueError("Both groups must contain at least one GraphMetrics item")

        dfA = pd.DataFrame([g.model_dump() for g in group_A])
        dfB = pd.DataFrame([g.model_dump() for g in group_B])

        # Ensure numeric and handle None
        metrics = [
            "diameter",
            "max_degree",
            "clustering_coefficient",
            "betweeness",
            "closeness",
            "page_rank",
            "average_relationship_score",
            "protagonist_count",
            "antagonist_count",
        ]

        statsA = (
            dfA[metrics]
            .apply(lambda s: pd.to_numeric(s, errors="coerce"))
            .agg(["mean", "min", "max", "std"])
            .T
        )
        statsB = (
            dfB[metrics]
            .apply(lambda s: pd.to_numeric(s, errors="coerce"))
            .agg(["mean", "min", "max", "std"])
            .T
        )

        config = get_config()
        out_dir = Path(config.notebooks_directory) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            a_mean = statsA.at[metric, "mean"]
            a_std = (
                statsA.at[metric, "std"]
                if not pd.isna(statsA.at[metric, "std"])
                else 0.0
            )
            a_min = statsA.at[metric, "min"]
            a_max = statsA.at[metric, "max"]

            b_mean = statsB.at[metric, "mean"]
            b_std = (
                statsB.at[metric, "std"]
                if not pd.isna(statsB.at[metric, "std"])
                else 0.0
            )
            b_min = statsB.at[metric, "min"]
            b_max = statsB.at[metric, "max"]

            ax.bar(
                [0, 1],
                [a_mean, b_mean],
                yerr=[a_std, b_std],
                color=["C0", "C1"],
                capsize=5,
            )
            # plot min/max as error markers
            ax.scatter([0, 1], [a_min, b_min], marker="_", color="black")
            ax.scatter([0, 1], [a_max, b_max], marker="_", color="black")
            ax.set_xticks([0, 1])
            ax.set_xticklabels([group_A_name, group_B_name], rotation=20)
            ax.set_title(metric)

        # hide any extra axes
        for j in range(len(metrics), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        safe_a = group_A_name.replace(" ", "_")
        safe_b = group_B_name.replace(" ", "_")
        out_path = out_dir / f"compare_{safe_a}_vs_{safe_b}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
