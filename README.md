# Taxonomy Creator

Automated taxonomy and relationship extraction from unstructured text using LLMs. Inspired by obsidian-wiki's emergent knowledge graph approach. Feed it documents, get structured taxonomies, concepts, entities, and their relationships -- no pre-built ontology required.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your API key
python main.py estimate --input data/input.xlsx
python main.py full-pipeline --input data/input.xlsx --dry-run
python main.py full-pipeline --input data/input.xlsx
```

## Commands

| Command | Description |
|---------|-------------|
| `extract` | Extract concepts, entities, relationships from input data |
| `build-taxonomy` | Build hierarchical taxonomy from extraction results |
| `cross-link` | Discover missing relationships across the taxonomy |
| `export` | Export to JSON, GraphML, Markdown, HTML, Obsidian vault |
| `full-pipeline` | Run all stages end-to-end |
| `estimate` | Estimate token usage and API costs without running |

## Flags

- `--dry-run` : Run everything except actual LLM calls. Shows prompts and cost estimates.
- `--batch-size N` : Override the batch size from config.
- `--input PATH` : Input file or directory.
- `--format FORMAT` : Export format (json, graphml, markdown, html, obsidian, all).
