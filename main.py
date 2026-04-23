#!/usr/bin/env python3
"""
Taxonomy Creator - Automated taxonomy extraction from unstructured text.

Usage:
  python main.py extract --input data/input.xlsx [--dry-run] [--batch-size 25]
  python main.py build-taxonomy --input output/extractions.json [--dry-run]
  python main.py cross-link --input output/taxonomy.json [--dry-run]
  python main.py export --input output/ --format all
  python main.py full-pipeline --input data/input.xlsx [--dry-run]
  python main.py estimate --input data/input.xlsx
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from src.config import Config
from src.exporter import TaxonomyExporter
from src.extractor import TaxonomyExtractor
from src.graph import KnowledgeGraph
from src.loader import DataLoader
from src.taxonomy_builder import TaxonomyBuilder

console = Console()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace, config: Config) -> None:
    """Run Stage 1: Extract concepts, entities, and relationships from input data."""
    console.print(Panel("[bold]Stage 1: Concept and Entity Extraction[/bold]", border_style="blue"))

    loader = DataLoader(config)
    loader.load(args.input)

    stats = loader.get_stats()
    _print_data_stats(stats)

    batches = loader.get_batches(batch_size=args.batch_size)

    extractor = TaxonomyExtractor(config)
    extractor.loader = loader  # share the loader for formatting
    extractor.extract_all(batches, dry_run=args.dry_run)


def cmd_build_taxonomy(args: argparse.Namespace, config: Config) -> None:
    """Run Stage 2: Build hierarchical taxonomy from extraction results."""
    console.print(Panel("[bold]Stage 2: Taxonomy Construction[/bold]", border_style="blue"))

    input_path = Path(args.input)
    if input_path.is_dir():
        input_path = input_path / "extractions.json"

    if not input_path.exists():
        console.print(f"[red]Extraction results not found at: {input_path}[/red]")
        console.print("[dim]Run 'extract' first, or specify the correct path.[/dim]")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        extractions = json.load(f)

    builder = TaxonomyBuilder(config)

    # Deduplicate
    concepts, merge_log = builder.deduplicate_concepts(extractions.get("concepts", []))
    entities = builder.deduplicate_entities(extractions.get("entities", []))

    n_results = extractions.get("metadata", {}).get("total_batches", 0) * config.batch_size

    # Build taxonomy
    taxonomy = builder.build_taxonomy(
        concepts=concepts,
        entities=entities,
        n_results=n_results,
        dry_run=args.dry_run,
    )

    if not args.dry_run and taxonomy:
        # Also save the deduplicated concepts and entities
        dedup_path = Path(config.output_dir) / "deduplicated.json"
        with open(dedup_path, "w", encoding="utf-8") as f:
            json.dump({
                "concepts": concepts,
                "entities": entities,
                "merge_log": merge_log,
            }, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Deduplicated data saved to:[/green] {dedup_path}")


def cmd_cross_link(args: argparse.Namespace, config: Config) -> None:
    """Run Stage 3: Discover missing cross-domain relationships."""
    console.print(Panel("[bold]Stage 3: Cross-Linking and Relationship Enrichment[/bold]", border_style="blue"))

    input_dir = Path(args.input)
    if input_dir.is_file():
        input_dir = input_dir.parent

    taxonomy_path = input_dir / "taxonomy.json"
    extractions_path = input_dir / "extractions.json"
    dedup_path = input_dir / "deduplicated.json"

    if not taxonomy_path.exists():
        console.print(f"[red]Taxonomy not found at: {taxonomy_path}[/red]")
        console.print("[dim]Run 'build-taxonomy' first.[/dim]")
        sys.exit(1)

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    relationships: list = []
    entities: list = []

    if extractions_path.exists():
        with open(extractions_path, "r", encoding="utf-8") as f:
            extractions = json.load(f)
        relationships = extractions.get("relationships", [])

    if dedup_path.exists():
        with open(dedup_path, "r", encoding="utf-8") as f:
            dedup = json.load(f)
        entities = dedup.get("entities", [])
    elif extractions_path.exists():
        entities = extractions.get("entities", [])

    n_results = 0
    if extractions_path.exists():
        meta = extractions.get("metadata", {})
        n_results = meta.get("total_batches", 0) * config.batch_size

    builder = TaxonomyBuilder(config)
    builder.cross_link(
        taxonomy=taxonomy,
        relationships=relationships,
        entities=entities,
        n_results=n_results,
        dry_run=args.dry_run,
    )


def cmd_export(args: argparse.Namespace, config: Config) -> None:
    """Export results to various formats."""
    console.print(Panel("[bold]Export[/bold]", border_style="blue"))

    input_dir = Path(args.input)
    if input_dir.is_file():
        input_dir = input_dir.parent

    export_format = args.format or "all"
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load available data
    taxonomy_data: dict = {}
    taxonomy_tree: list = []
    extractions: dict = {}
    crosslinks: dict = {}
    dedup_data: dict = {}

    taxonomy_path = input_dir / "taxonomy.json"
    extractions_path = input_dir / "extractions.json"
    crosslinks_path = input_dir / "crosslinks.json"
    dedup_path = input_dir / "deduplicated.json"

    if taxonomy_path.exists():
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy_data = json.load(f)
        taxonomy_tree = taxonomy_data.get("taxonomy", [])

    if extractions_path.exists():
        with open(extractions_path, "r", encoding="utf-8") as f:
            extractions = json.load(f)

    if crosslinks_path.exists():
        with open(crosslinks_path, "r", encoding="utf-8") as f:
            crosslinks = json.load(f)

    if dedup_path.exists():
        with open(dedup_path, "r", encoding="utf-8") as f:
            dedup_data = json.load(f)

    # Gather all data
    concepts = dedup_data.get("concepts", extractions.get("concepts", []))
    entities = dedup_data.get("entities", extractions.get("entities", []))
    relationships = extractions.get("relationships", [])
    new_rels = crosslinks.get("new_relationships", [])
    all_relationships = relationships + new_rels

    # Build the knowledge graph
    graph = KnowledgeGraph()
    graph.add_concepts(concepts)
    graph.add_entities(entities)
    graph.add_relationships(all_relationships)
    if taxonomy_tree:
        graph.add_taxonomy_edges(taxonomy_tree)

    analysis = graph.analyze()

    # Export based on format
    formats_to_export = (
        ["json", "graphml", "markdown", "html", "obsidian"]
        if export_format == "all"
        else [export_format]
    )

    for fmt in formats_to_export:
        if fmt == "json":
            graph.export_json(str(output_dir / "knowledge_graph.json"))

        elif fmt == "graphml":
            graph.export_graphml(str(output_dir / "knowledge_graph.graphml"))

        elif fmt == "markdown":
            graph.export_markdown(str(output_dir / "knowledge_graph.md"))
            if taxonomy_tree:
                TaxonomyExporter.export_taxonomy_markdown(
                    taxonomy_tree, str(output_dir / "taxonomy_tree.md")
                )
            if all_relationships:
                TaxonomyExporter.export_relationships_csv(
                    all_relationships, str(output_dir / "relationships.csv")
                )

        elif fmt == "html":
            graph.export_html(str(output_dir / "knowledge_graph.html"))
            TaxonomyExporter.export_summary_report(
                taxonomy=taxonomy_tree,
                analysis=analysis,
                concepts=concepts,
                entities=entities,
                relationships=all_relationships,
                path=str(output_dir / "summary_report.html"),
            )

        elif fmt == "obsidian":
            TaxonomyExporter.export_obsidian_vault(
                taxonomy=taxonomy_tree,
                relationships=all_relationships,
                entities=entities,
                vault_path=str(output_dir / "obsidian_vault"),
            )

    console.print("\n[bold green]Export complete.[/bold green]")


def cmd_full_pipeline(args: argparse.Namespace, config: Config) -> None:
    """Run the complete pipeline: extract -> build-taxonomy -> cross-link -> export."""
    console.print(Panel(
        "[bold]Full Pipeline[/bold]\n"
        "Stage 1: Extract  ->  Stage 2: Taxonomy  ->  Stage 3: Cross-Link  ->  Export",
        border_style="magenta",
    ))

    # Stage 1: Extract
    console.print("\n" + "=" * 60)
    cmd_extract(args, config)

    if args.dry_run:
        console.print("\n[yellow]Dry run complete. Review the prompts in output/dry_run/ and rerun without --dry-run.[/yellow]")
        return

    # Stage 2: Build taxonomy
    console.print("\n" + "=" * 60)
    tax_args = argparse.Namespace(
        input=str(Path(config.output_dir) / "extractions.json"),
        dry_run=False,
        batch_size=args.batch_size,
    )
    cmd_build_taxonomy(tax_args, config)

    # Stage 3: Cross-link
    console.print("\n" + "=" * 60)
    cl_args = argparse.Namespace(
        input=config.output_dir,
        dry_run=False,
        batch_size=args.batch_size,
    )
    cmd_cross_link(cl_args, config)

    # Export all formats
    console.print("\n" + "=" * 60)
    exp_args = argparse.Namespace(
        input=config.output_dir,
        format="all",
        dry_run=False,
        batch_size=args.batch_size,
    )
    cmd_export(exp_args, config)


def cmd_estimate(args: argparse.Namespace, config: Config) -> None:
    """Estimate token usage and API costs without running anything."""
    console.print(Panel("[bold]Cost Estimation[/bold]", border_style="blue"))

    loader = DataLoader(config)
    loader.load(args.input)
    stats = loader.get_stats()
    _print_data_stats(stats)

    pricing = config.get_model_pricing()
    batch_size = args.batch_size or config.batch_size
    n_batches = stats["total_batches"]

    # Import the prompt template to estimate its size
    from src.prompts import CROSSLINK_PROMPT, EXTRACTION_PROMPT, TAXONOMY_PROMPT

    # Stage 1: Extraction
    avg_batch_data_tokens = stats["avg_chars_per_row"] * batch_size / 4
    prompt_template_tokens = len(EXTRACTION_PROMPT) / 4
    stage1_input_tokens = int((avg_batch_data_tokens + prompt_template_tokens) * n_batches)
    stage1_output_tokens = int(n_batches * 2000)  # ~2K tokens per response

    # Stage 2: Taxonomy (one call with all concepts)
    estimated_concepts = int(n_batches * 15)  # ~15 unique concepts per batch
    stage2_input_tokens = int(len(TAXONOMY_PROMPT) / 4 + estimated_concepts * 50)
    stage2_output_tokens = 4000

    # Stage 3: Cross-linking (one call)
    stage3_input_tokens = int(len(CROSSLINK_PROMPT) / 4 + 10000)  # taxonomy + relationships
    stage3_output_tokens = 3000

    total_input = stage1_input_tokens + stage2_input_tokens + stage3_input_tokens
    total_output = stage1_output_tokens + stage2_output_tokens + stage3_output_tokens

    total_input_cost = (total_input / 1_000_000) * pricing["input"]
    total_output_cost = (total_output / 1_000_000) * pricing["output"]

    # Print cost breakdown
    table = Table(title=f"Cost Estimate - {config.model_name}", border_style="blue")
    table.add_column("Stage", style="bold")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Input Cost", justify="right")
    table.add_column("Output Cost", justify="right")
    table.add_column("Total Cost", justify="right", style="bold")

    def _row(name: str, inp: int, out: int) -> None:
        ic = (inp / 1_000_000) * pricing["input"]
        oc = (out / 1_000_000) * pricing["output"]
        table.add_row(name, f"{inp:,}", f"{out:,}", f"${ic:.4f}", f"${oc:.4f}", f"${ic + oc:.4f}")

    _row("1. Extraction", stage1_input_tokens, stage1_output_tokens)
    _row("2. Taxonomy", stage2_input_tokens, stage2_output_tokens)
    _row("3. Cross-linking", stage3_input_tokens, stage3_output_tokens)
    table.add_section()
    _row("TOTAL", total_input, total_output)

    console.print()
    console.print(table)

    console.print(f"\n[bold]Pricing used:[/bold] ${pricing['input']}/M input, ${pricing['output']}/M output")
    console.print(f"[bold]Total estimated cost:[/bold] [green]${total_input_cost + total_output_cost:.4f}[/green]")
    console.print(f"[bold]Estimated API calls:[/bold] {n_batches + 2}")

    console.print("\n[dim]These are rough estimates. Actual costs depend on content complexity and LLM response verbosity.[/dim]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_data_stats(stats: dict) -> None:
    """Print data statistics in a table."""
    table = Table(title="Input Data Statistics", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total rows", f"{stats['total_rows']:,}")
    table.add_row("Avg title length", f"{stats['avg_title_length']:.0f} chars")
    table.add_row("Avg description length", f"{stats['avg_description_length']:.0f} chars")
    table.add_row("Total characters", f"{stats['total_characters']:,}")
    table.add_row("Avg chars/row", f"{stats['avg_chars_per_row']:.0f}")
    table.add_row("Estimated tokens (total)", f"{stats['estimated_tokens']:,}")
    table.add_row("Batch size", f"{stats['batch_size']}")
    table.add_row("Total batches", f"{stats['total_batches']}")
    console.print(table)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Taxonomy Creator - Automated taxonomy extraction from unstructured text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract
    p_extract = subparsers.add_parser("extract", help="Extract concepts, entities, relationships")
    p_extract.add_argument("--input", "-i", type=str, required=True, help="Input Excel/CSV file path")
    p_extract.add_argument("--dry-run", action="store_true", help="Show prompts and costs without API calls")
    p_extract.add_argument("--batch-size", type=int, default=None, help="Override batch size")

    # Build taxonomy
    p_taxonomy = subparsers.add_parser("build-taxonomy", help="Build taxonomy from extractions")
    p_taxonomy.add_argument("--input", "-i", type=str, required=True, help="Path to extractions.json or output dir")
    p_taxonomy.add_argument("--dry-run", action="store_true", help="Show prompts and costs without API calls")
    p_taxonomy.add_argument("--batch-size", type=int, default=None)

    # Cross-link
    p_crosslink = subparsers.add_parser("cross-link", help="Discover missing relationships")
    p_crosslink.add_argument("--input", "-i", type=str, required=True, help="Path to taxonomy.json or output dir")
    p_crosslink.add_argument("--dry-run", action="store_true", help="Show prompts and costs without API calls")
    p_crosslink.add_argument("--batch-size", type=int, default=None)

    # Export
    p_export = subparsers.add_parser("export", help="Export to various formats")
    p_export.add_argument("--input", "-i", type=str, required=True, help="Path to output directory")
    p_export.add_argument("--format", "-f", type=str, default="all",
                          choices=["json", "graphml", "markdown", "html", "obsidian", "all"],
                          help="Export format")
    p_export.add_argument("--dry-run", action="store_true")
    p_export.add_argument("--batch-size", type=int, default=None)

    # Full pipeline
    p_full = subparsers.add_parser("full-pipeline", help="Run all stages end-to-end")
    p_full.add_argument("--input", "-i", type=str, required=True, help="Input Excel/CSV file path")
    p_full.add_argument("--dry-run", action="store_true", help="Show prompts and costs without API calls")
    p_full.add_argument("--batch-size", type=int, default=None, help="Override batch size")

    # Estimate
    p_estimate = subparsers.add_parser("estimate", help="Estimate token usage and costs")
    p_estimate.add_argument("--input", "-i", type=str, required=True, help="Input Excel/CSV file path")
    p_estimate.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p_estimate.add_argument("--dry-run", action="store_true")

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(verbose=getattr(args, "verbose", False))

    # Build config with overrides
    overrides = {}
    if hasattr(args, "input") and args.input and args.command in ("extract", "full-pipeline", "estimate"):
        overrides["input"] = {"file": args.input}

    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None:
        overrides.setdefault("input", {})["batch_size"] = batch_size

    config = Config(config_path=args.config, overrides=overrides)

    console.print(Panel(
        f"[bold]Taxonomy Creator[/bold]\n"
        f"Model: {config.model_name}  |  Provider: {config.model_provider}  |  Batch size: {config.batch_size}",
        border_style="magenta",
    ))

    command_map = {
        "extract": cmd_extract,
        "build-taxonomy": cmd_build_taxonomy,
        "cross-link": cmd_cross_link,
        "export": cmd_export,
        "full-pipeline": cmd_full_pipeline,
        "estimate": cmd_estimate,
    }

    handler = command_map.get(args.command)
    if handler:
        handler(args, config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
