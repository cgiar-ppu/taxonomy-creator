"""Knowledge graph construction and analysis using NetworkX."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from rich.console import Console
from rich.table import Table

from src.html_template import GRAPH_HTML_TEMPLATE

logger = logging.getLogger(__name__)
console = Console()

# Color scheme for node types
NODE_COLORS: Dict[str, str] = {
    "concept": "#4A90D9",
    "entity": "#E67E22",
    "organism": "#27AE60",
    "organization": "#8E44AD",
    "place": "#E74C3C",
    "tool": "#F39C12",
    "role": "#1ABC9C",
    "domain": "#2C3E50",
}

# Color scheme for relationship types
EDGE_COLORS: Dict[str, str] = {
    "is_a": "#3498DB",
    "part_of": "#9B59B6",
    "uses": "#E67E22",
    "produces": "#27AE60",
    "targets": "#E74C3C",
    "located_in": "#F39C12",
    "collaborates_with": "#1ABC9C",
    "addresses": "#E91E63",
    "related_to": "#95A5A6",
}


class KnowledgeGraph:
    """NetworkX-based knowledge graph for concepts, entities, and relationships."""

    def __init__(self) -> None:
        """Initialize an empty knowledge graph."""
        self.graph = nx.DiGraph()

    def add_concepts(self, concepts: List[Dict[str, Any]]) -> None:
        """Add concept nodes to the graph.

        Args:
            concepts: List of concept dicts with name, category, description, frequency.
        """
        for concept in concepts:
            name = concept.get("name", "").strip()
            if not name:
                continue

            self.graph.add_node(
                name,
                node_type="concept",
                category=concept.get("category", "approach"),
                description=concept.get("description", ""),
                frequency=concept.get("frequency", 1),
                aliases=concept.get("aliases", []),
                color=NODE_COLORS.get("concept", "#4A90D9"),
            )

    def add_entities(self, entities: List[Dict[str, Any]]) -> None:
        """Add entity nodes to the graph.

        Args:
            entities: List of entity dicts with name, type, description.
        """
        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue

            entity_type = entity.get("type", "entity")
            self.graph.add_node(
                name,
                node_type="entity",
                entity_type=entity_type,
                description=entity.get("description", ""),
                frequency=1,
                color=NODE_COLORS.get(entity_type, NODE_COLORS["entity"]),
            )

    def add_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        """Add relationship edges to the graph.

        Args:
            relationships: List of relationship dicts with source, type, target, confidence, evidence.
        """
        for rel in relationships:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            rel_type = rel.get("type", "related_to")

            if not source or not target:
                continue

            # Ensure both nodes exist (add as unknown if not)
            if source not in self.graph:
                self.graph.add_node(source, node_type="unknown", color="#CCCCCC", frequency=1)
            if target not in self.graph:
                self.graph.add_node(target, node_type="unknown", color="#CCCCCC", frequency=1)

            self.graph.add_edge(
                source,
                target,
                relationship_type=rel_type,
                confidence=rel.get("confidence", "inferred"),
                evidence=rel.get("evidence", ""),
                color=EDGE_COLORS.get(rel_type, EDGE_COLORS["related_to"]),
            )

    def add_taxonomy_edges(self, taxonomy: List[Dict[str, Any]]) -> None:
        """Add hierarchical edges from the taxonomy structure.

        Args:
            taxonomy: List of top-level taxonomy domain dicts with nested children.
        """
        def _walk(parent_name: str, children: List[Dict[str, Any]]) -> None:
            for child in children:
                child_name = child.get("name", "").strip()
                if not child_name:
                    continue

                if child_name not in self.graph:
                    self.graph.add_node(
                        child_name,
                        node_type="concept",
                        description=child.get("description", ""),
                        frequency=child.get("frequency", 1),
                        color=NODE_COLORS["concept"],
                    )

                self.graph.add_edge(
                    child_name,
                    parent_name,
                    relationship_type="is_a",
                    confidence="taxonomy",
                    color=EDGE_COLORS["is_a"],
                )

                if "children" in child:
                    _walk(child_name, child["children"])

        for domain in taxonomy:
            domain_name = domain.get("name", "").strip()
            if not domain_name:
                continue

            if domain_name not in self.graph:
                self.graph.add_node(
                    domain_name,
                    node_type="domain",
                    description=domain.get("description", ""),
                    frequency=0,
                    color=NODE_COLORS["domain"],
                )

            if "children" in domain:
                _walk(domain_name, domain["children"])

    def analyze(self) -> Dict[str, Any]:
        """Compute graph analytics: hubs, bridges, clusters, orphans.

        Returns:
            Dictionary with analysis results.
        """
        if len(self.graph) == 0:
            return {"error": "Empty graph"}

        # Basic stats
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()

        # Node type counts
        type_counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            type_counts[nt] = type_counts.get(nt, 0) + 1

        # Edge type counts
        edge_type_counts: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rt = data.get("relationship_type", "unknown")
            edge_type_counts[rt] = edge_type_counts.get(rt, 0) + 1

        # Hub nodes (highest degree)
        degree_map = dict(self.graph.degree())
        sorted_by_degree = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)
        hub_nodes = sorted_by_degree[:20]

        # Orphan nodes (degree 0)
        orphans = [node for node, deg in degree_map.items() if deg == 0]

        # Bridge nodes (high betweenness centrality) -- use undirected version
        undirected = self.graph.to_undirected()
        try:
            betweenness = nx.betweenness_centrality(undirected)
            sorted_by_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
            bridge_nodes = sorted_by_betweenness[:20]
        except Exception:
            bridge_nodes = []

        # Weakly connected components
        if nx.is_directed(self.graph):
            components = list(nx.weakly_connected_components(self.graph))
        else:
            components = list(nx.connected_components(undirected))

        cluster_sizes = sorted([len(c) for c in components], reverse=True)

        analysis = {
            "total_nodes": n_nodes,
            "total_edges": n_edges,
            "node_type_counts": type_counts,
            "edge_type_counts": edge_type_counts,
            "hub_nodes": [{"name": n, "degree": d} for n, d in hub_nodes],
            "bridge_nodes": [{"name": n, "centrality": round(c, 4)} for n, c in bridge_nodes],
            "orphan_nodes": orphans[:50],  # limit output
            "orphan_count": len(orphans),
            "connected_components": len(components),
            "largest_component_size": cluster_sizes[0] if cluster_sizes else 0,
            "component_sizes": cluster_sizes[:20],
        }

        self._print_analysis(analysis)
        return analysis

    def _print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print graph analysis as rich tables."""
        # Summary table
        summary = Table(title="Knowledge Graph Summary", border_style="blue")
        summary.add_column("Metric", style="bold")
        summary.add_column("Value", justify="right")
        summary.add_row("Total nodes", f"{analysis['total_nodes']:,}")
        summary.add_row("Total edges", f"{analysis['total_edges']:,}")
        summary.add_row("Connected components", f"{analysis['connected_components']:,}")
        summary.add_row("Largest component", f"{analysis['largest_component_size']:,}")
        summary.add_row("Orphan nodes", f"{analysis['orphan_count']:,}")
        console.print(summary)

        # Node types
        nt_table = Table(title="Node Types", border_style="cyan")
        nt_table.add_column("Type", style="bold")
        nt_table.add_column("Count", justify="right")
        for nt, count in sorted(analysis["node_type_counts"].items()):
            nt_table.add_row(nt, f"{count:,}")
        console.print(nt_table)

        # Top hubs
        hub_table = Table(title="Top Hub Nodes (by degree)", border_style="green")
        hub_table.add_column("Node", style="bold")
        hub_table.add_column("Degree", justify="right")
        for hub in analysis["hub_nodes"][:10]:
            hub_table.add_row(hub["name"], str(hub["degree"]))
        console.print(hub_table)

    def export_json(self, path: str) -> None:
        """Export graph in node-link JSON format.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        console.print(f"[green]Graph JSON saved to:[/green] {path}")

    def export_graphml(self, path: str) -> None:
        """Export graph in GraphML format for Gephi/Cytoscape.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # GraphML does not support list attributes; convert them to strings
        graph_copy = self.graph.copy()
        for node in graph_copy.nodes():
            for key, val in list(graph_copy.nodes[node].items()):
                if isinstance(val, (list, dict)):
                    graph_copy.nodes[node][key] = json.dumps(val)

        nx.write_graphml(graph_copy, path)
        console.print(f"[green]GraphML saved to:[/green] {path}")

    def export_markdown(self, path: str) -> None:
        """Export graph as a human-readable Markdown taxonomy tree.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        lines = ["# Knowledge Graph - Taxonomy View\n"]

        # Group nodes by type
        concepts = []
        entities_by_type: Dict[str, List[str]] = {}

        for node, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            if nt in ("concept", "domain"):
                concepts.append((node, data))
            elif nt == "entity":
                et = data.get("entity_type", "other")
                entities_by_type.setdefault(et, []).append(node)

        # Print concepts sorted by frequency
        lines.append("## Concepts\n")
        concepts.sort(key=lambda x: x[1].get("frequency", 0), reverse=True)
        for name, data in concepts:
            freq = data.get("frequency", 0)
            desc = data.get("description", "")
            lines.append(f"- **{name}** (freq: {freq}) - {desc}")

        # Print entities by type
        lines.append("\n## Entities\n")
        for etype, names in sorted(entities_by_type.items()):
            lines.append(f"### {etype.title()}\n")
            for name in sorted(names):
                lines.append(f"- {name}")
            lines.append("")

        # Print relationships
        lines.append("## Relationships\n")
        for source, target, data in self.graph.edges(data=True):
            rt = data.get("relationship_type", "related_to")
            conf = data.get("confidence", "")
            lines.append(f"- {source} --[{rt}]--> {target} ({conf})")

        text = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        console.print(f"[green]Markdown export saved to:[/green] {path}")

    def export_html(self, path: str) -> None:
        """Export an interactive HTML visualization using vis.js.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Build node and edge lists for vis.js
        nodes_js = []
        for node, data in self.graph.nodes(data=True):
            freq = data.get("frequency", 1)
            size = max(10, min(50, 10 + freq * 2))
            node_type = data.get("node_type", "unknown")
            entity_type = data.get("entity_type", "")
            color = data.get("color", "#CCCCCC")
            description = data.get("description", "").replace('"', '\\"').replace("\n", " ")

            label_type = entity_type if entity_type else node_type
            title_text = f"{node}\\nType: {label_type}\\nFrequency: {freq}\\n{description}"

            nodes_js.append({
                "id": node,
                "label": node,
                "title": title_text,
                "size": size,
                "color": color,
                "group": label_type,
                "font": {"size": max(8, min(16, 8 + freq))},
            })

        edges_js = []
        for source, target, data in self.graph.edges(data=True):
            rel_type = data.get("relationship_type", "related_to")
            color = data.get("color", "#95A5A6")
            confidence = data.get("confidence", "inferred")

            width = 1
            if confidence == "extracted":
                width = 2
            elif confidence == "taxonomy":
                width = 3

            edges_js.append({
                "from": source,
                "to": target,
                "label": rel_type,
                "color": {"color": color, "opacity": 0.7},
                "width": width,
                "arrows": "to",
                "title": f"{rel_type} ({confidence})",
                "font": {"size": 8, "align": "middle"},
            })

        # Build legend data
        legend_items = []
        for label, color in NODE_COLORS.items():
            legend_items.append({"label": label, "color": color, "type": "node"})
        for label, color in EDGE_COLORS.items():
            legend_items.append({"label": label, "color": color, "type": "edge"})

        # Render HTML
        nodes_json = json.dumps(nodes_js, ensure_ascii=False)
        edges_json = json.dumps(edges_js, ensure_ascii=False)
        legend_json = json.dumps(legend_items, ensure_ascii=False)

        html_content = GRAPH_HTML_TEMPLATE.format(
            nodes_json=nodes_json,
            edges_json=edges_json,
            legend_json=legend_json,
            total_nodes=len(nodes_js),
            total_edges=len(edges_js),
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
        console.print(f"[green]Interactive HTML graph saved to:[/green] {path}")
