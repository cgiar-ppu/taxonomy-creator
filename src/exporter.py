"""Export utilities for taxonomy, relationships, and Obsidian vault generation."""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


class TaxonomyExporter:
    """Exports taxonomy and graph data in multiple formats."""

    @staticmethod
    def export_taxonomy_markdown(taxonomy: List[Dict[str, Any]], path: str) -> None:
        """Export taxonomy as an indented Markdown tree with descriptions.

        Args:
            taxonomy: List of top-level taxonomy domain dicts with nested children.
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Taxonomy\n"]

        def _walk(nodes: List[Dict[str, Any]], depth: int) -> None:
            for node in nodes:
                name = node.get("name", "Unknown")
                desc = node.get("description", "")
                freq = node.get("frequency", "")
                aliases = node.get("aliases", [])

                indent = "  " * depth
                freq_str = f" (freq: {freq})" if freq else ""
                alias_str = f" [aliases: {', '.join(aliases)}]" if aliases else ""

                if depth == 0:
                    lines.append(f"\n## {name}")
                    if desc:
                        lines.append(f"_{desc}_\n")
                elif depth == 1:
                    lines.append(f"\n### {name}")
                    if desc:
                        lines.append(f"_{desc}_\n")
                else:
                    lines.append(f"{indent}- **{name}**{freq_str}{alias_str}")
                    if desc and depth <= 2:
                        lines.append(f"{indent}  _{desc}_")

                if "children" in node:
                    _walk(node["children"], depth + 1)

        _walk(taxonomy, 0)

        text = "\n".join(lines) + "\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        console.print(f"[green]Taxonomy Markdown saved to:[/green] {path}")

    @staticmethod
    def export_relationships_csv(relationships: List[Dict[str, Any]], path: str) -> None:
        """Export relationships as a CSV file.

        Args:
            relationships: List of relationship dicts.
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["source", "type", "target", "confidence", "evidence"]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rel in relationships:
                writer.writerow({
                    "source": rel.get("source", ""),
                    "type": rel.get("type", "related_to"),
                    "target": rel.get("target", ""),
                    "confidence": rel.get("confidence", ""),
                    "evidence": rel.get("evidence", ""),
                })

        console.print(f"[green]Relationships CSV saved to:[/green] {path}")

    @staticmethod
    def export_summary_report(
        taxonomy: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        concepts: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        path: str,
    ) -> None:
        """Export an HTML summary report with statistics and tables.

        Args:
            taxonomy: The taxonomy tree structure.
            analysis: Graph analysis results from KnowledgeGraph.analyze().
            concepts: Deduplicated concept list.
            entities: Deduplicated entity list.
            relationships: All relationships.
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Sort concepts by frequency
        top_concepts = sorted(concepts, key=lambda x: x.get("frequency", 0), reverse=True)[:30]

        # Count entities by type
        entity_type_counts: Dict[str, int] = {}
        for ent in entities:
            et = ent.get("type", "other")
            entity_type_counts[et] = entity_type_counts.get(et, 0) + 1

        # Count relationships by type
        rel_type_counts: Dict[str, int] = {}
        for rel in relationships:
            rt = rel.get("type", "related_to")
            rel_type_counts[rt] = rel_type_counts.get(rt, 0) + 1

        # Build concept rows HTML
        concept_rows = ""
        for c in top_concepts:
            concept_rows += (
                f"<tr><td>{c.get('name', '')}</td>"
                f"<td>{c.get('category', '')}</td>"
                f"<td>{c.get('frequency', 0)}</td>"
                f"<td>{c.get('description', '')}</td></tr>\n"
            )

        # Build entity type rows
        entity_rows = ""
        for et, count in sorted(entity_type_counts.items(), key=lambda x: x[1], reverse=True):
            entity_rows += f"<tr><td>{et}</td><td>{count}</td></tr>\n"

        # Build relationship type rows
        rel_rows = ""
        for rt, count in sorted(rel_type_counts.items(), key=lambda x: x[1], reverse=True):
            rel_rows += f"<tr><td>{rt}</td><td>{count}</td></tr>\n"

        # Build hub nodes rows
        hub_rows = ""
        for hub in analysis.get("hub_nodes", [])[:15]:
            hub_rows += f"<tr><td>{hub['name']}</td><td>{hub['degree']}</td></tr>\n"

        # Build taxonomy outline
        taxonomy_html = ""

        def _tax_html(nodes: List[Dict[str, Any]], depth: int) -> str:
            result = "<ul>"
            for node in nodes:
                freq = node.get("frequency", "")
                freq_badge = f' <span class="badge">{freq}</span>' if freq else ""
                result += f"<li><strong>{node.get('name', '')}</strong>{freq_badge}"
                if node.get("description"):
                    result += f" -- {node['description']}"
                if "children" in node and node["children"]:
                    result += _tax_html(node["children"], depth + 1)
                result += "</li>"
            result += "</ul>"
            return result

        taxonomy_html = _tax_html(taxonomy, 0)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Taxonomy Creator - Summary Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1100px;
            margin: 0 auto;
            padding: 20px 40px;
            background: #fafafa;
            color: #333;
            line-height: 1.6;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e94560; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .stat-card .number {{
            font-size: 32px;
            font-weight: bold;
            color: #e94560;
        }}
        .stat-card .label {{
            font-size: 13px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            background: white;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        th {{
            background: #2c3e50;
            color: white;
            padding: 10px 14px;
            text-align: left;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 8px 14px;
            border-bottom: 1px solid #eee;
            font-size: 14px;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .badge {{
            background: #e94560;
            color: white;
            padding: 1px 8px;
            border-radius: 10px;
            font-size: 11px;
            margin-left: 4px;
        }}
        .taxonomy-tree ul {{
            list-style-type: none;
            padding-left: 20px;
        }}
        .taxonomy-tree > ul {{
            padding-left: 0;
        }}
        .taxonomy-tree li {{
            margin: 4px 0;
            font-size: 14px;
        }}
        .taxonomy-tree li strong {{
            color: #2c3e50;
        }}
        footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #999;
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>Taxonomy Creator - Summary Report</h1>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="number">{len(concepts):,}</div>
            <div class="label">Unique Concepts</div>
        </div>
        <div class="stat-card">
            <div class="number">{len(entities):,}</div>
            <div class="label">Unique Entities</div>
        </div>
        <div class="stat-card">
            <div class="number">{len(relationships):,}</div>
            <div class="label">Relationships</div>
        </div>
        <div class="stat-card">
            <div class="number">{analysis.get('total_nodes', 0):,}</div>
            <div class="label">Graph Nodes</div>
        </div>
        <div class="stat-card">
            <div class="number">{analysis.get('total_edges', 0):,}</div>
            <div class="label">Graph Edges</div>
        </div>
        <div class="stat-card">
            <div class="number">{analysis.get('connected_components', 0):,}</div>
            <div class="label">Components</div>
        </div>
    </div>

    <h2>Top Concepts (by frequency)</h2>
    <table>
        <thead>
            <tr><th>Concept</th><th>Category</th><th>Frequency</th><th>Description</th></tr>
        </thead>
        <tbody>
            {concept_rows}
        </tbody>
    </table>

    <h2>Entity Types</h2>
    <table>
        <thead>
            <tr><th>Entity Type</th><th>Count</th></tr>
        </thead>
        <tbody>
            {entity_rows}
        </tbody>
    </table>

    <h2>Relationship Types</h2>
    <table>
        <thead>
            <tr><th>Relationship Type</th><th>Count</th></tr>
        </thead>
        <tbody>
            {rel_rows}
        </tbody>
    </table>

    <h2>Hub Nodes (most connected)</h2>
    <table>
        <thead>
            <tr><th>Node</th><th>Degree</th></tr>
        </thead>
        <tbody>
            {hub_rows}
        </tbody>
    </table>

    <h2>Taxonomy Tree</h2>
    <div class="taxonomy-tree">
        {taxonomy_html}
    </div>

    <footer>
        Generated by Taxonomy Creator
    </footer>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        console.print(f"[green]Summary report saved to:[/green] {path}")

    @staticmethod
    def export_obsidian_vault(
        taxonomy: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        vault_path: str,
    ) -> None:
        """Generate an Obsidian-compatible Markdown vault with wikilinks.

        Creates one .md file per concept/entity, with wikilinks to related nodes.

        Args:
            taxonomy: The taxonomy tree structure.
            relationships: All relationships.
            entities: Deduplicated entity list.
            vault_path: Root directory for the Obsidian vault.
        """
        vault = Path(vault_path)
        vault.mkdir(parents=True, exist_ok=True)

        # Build a lookup of relationships by source and target
        rels_by_source: Dict[str, List[Dict[str, Any]]] = {}
        rels_by_target: Dict[str, List[Dict[str, Any]]] = {}
        for rel in relationships:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if src:
                rels_by_source.setdefault(src, []).append(rel)
            if tgt:
                rels_by_target.setdefault(tgt, []).append(rel)

        # Collect all concepts from taxonomy
        all_concepts: List[Dict[str, Any]] = []

        def _collect(nodes: List[Dict[str, Any]], parent: Optional[str] = None) -> None:
            for node in nodes:
                concept = {
                    "name": node.get("name", ""),
                    "description": node.get("description", ""),
                    "frequency": node.get("frequency", ""),
                    "aliases": node.get("aliases", []),
                    "parent": parent,
                }
                all_concepts.append(concept)
                if "children" in node:
                    _collect(node["children"], node.get("name"))

        _collect(taxonomy)

        # Create concept pages
        concepts_dir = vault / "Concepts"
        concepts_dir.mkdir(exist_ok=True)

        for concept in all_concepts:
            name = concept["name"]
            safe_name = _safe_filename(name)
            lines = [f"# {name}\n"]

            if concept["description"]:
                lines.append(f"{concept['description']}\n")

            if concept["aliases"]:
                lines.append(f"**Aliases:** {', '.join(concept['aliases'])}\n")

            if concept["frequency"]:
                lines.append(f"**Frequency:** {concept['frequency']}\n")

            if concept["parent"]:
                lines.append(f"**Parent:** [[{concept['parent']}]]\n")

            # Outgoing relationships
            out_rels = rels_by_source.get(name, []) + rels_by_source.get(name.lower(), [])
            if out_rels:
                lines.append("## Relationships\n")
                for rel in out_rels:
                    lines.append(f"- {rel.get('type', 'related_to')} [[{rel.get('target', '')}]]")

            # Incoming relationships
            in_rels = rels_by_target.get(name, []) + rels_by_target.get(name.lower(), [])
            if in_rels:
                lines.append("\n## Referenced By\n")
                for rel in in_rels:
                    lines.append(f"- [[{rel.get('source', '')}]] {rel.get('type', 'related_to')} this")

            file_path = concepts_dir / f"{safe_name}.md"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        # Create entity pages
        entities_dir = vault / "Entities"
        entities_dir.mkdir(exist_ok=True)

        for entity in entities:
            name = entity.get("name", "")
            if not name:
                continue

            safe_name = _safe_filename(name)
            lines = [f"# {name}\n"]

            etype = entity.get("type", "")
            if etype:
                lines.append(f"**Type:** {etype}\n")

            desc = entity.get("description", "")
            if desc:
                lines.append(f"{desc}\n")

            out_rels = rels_by_source.get(name, [])
            if out_rels:
                lines.append("## Relationships\n")
                for rel in out_rels:
                    lines.append(f"- {rel.get('type', 'related_to')} [[{rel.get('target', '')}]]")

            in_rels = rels_by_target.get(name, [])
            if in_rels:
                lines.append("\n## Referenced By\n")
                for rel in in_rels:
                    lines.append(f"- [[{rel.get('source', '')}]] {rel.get('type', 'related_to')} this")

            file_path = entities_dir / f"{safe_name}.md"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

        # Create an index page
        index_lines = ["# Knowledge Graph Index\n"]
        index_lines.append("## Taxonomy\n")
        for concept in all_concepts:
            if concept["parent"] is None:
                index_lines.append(f"- [[{concept['name']}]]")
        index_lines.append("\n## Entities\n")
        entity_types: Dict[str, List[str]] = {}
        for ent in entities:
            et = ent.get("type", "other")
            entity_types.setdefault(et, []).append(ent.get("name", ""))
        for et, names in sorted(entity_types.items()):
            index_lines.append(f"\n### {et.title()}\n")
            for name in sorted(names):
                index_lines.append(f"- [[{name}]]")

        index_path = vault / "Index.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(index_lines) + "\n")

        total_files = len(all_concepts) + len(entities) + 1
        console.print(f"[green]Obsidian vault saved to:[/green] {vault_path} ({total_files} files)")


def _safe_filename(name: str) -> str:
    """Convert a name to a filesystem-safe filename.

    Args:
        name: The concept or entity name.

    Returns:
        Sanitized string safe for use as a filename (without extension).
    """
    # Replace characters that are problematic in filenames
    replacements = {
        "/": "-",
        "\\": "-",
        ":": "-",
        "*": "",
        "?": "",
        '"': "",
        "<": "",
        ">": "",
        "|": "",
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    # Truncate to a reasonable length
    return result[:100].strip()
