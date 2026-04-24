#!/usr/bin/env python3
"""
Taxonomy Creator - Flask Web Application

A local web UI for interacting with the taxonomy extraction pipeline.
Start with: python app.py
"""

import io
import json
import logging
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    session,
)

from src.config import Config, MODEL_PRICING
from src.exporter import TaxonomyExporter
from src.extractor import TaxonomyExtractor
from src.graph import KnowledgeGraph
from src.loader import DataLoader
from src.prompts import CROSSLINK_PROMPT, EXTRACTION_PROMPT, TAXONOMY_PROMPT
from src.taxonomy_builder import TaxonomyBuilder

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.urandom(32)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("taxonomy-web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Pipeline status -- shared state for background thread
# ---------------------------------------------------------------------------

pipeline_status = {
    "stage": "idle",
    "progress": 0,
    "total": 0,
    "message": "",
    "error": None,
    "results": None,
    "logs": [],
    "started_at": None,
    "finished_at": None,
}

_pipeline_lock = threading.Lock()


def _reset_status():
    """Reset pipeline status to idle."""
    global pipeline_status
    with _pipeline_lock:
        pipeline_status = {
            "stage": "idle",
            "progress": 0,
            "total": 0,
            "message": "",
            "error": None,
            "results": None,
            "logs": [],
            "started_at": None,
            "finished_at": None,
        }


def _log(msg):
    """Append a log message to the pipeline status."""
    with _pipeline_lock:
        pipeline_status["logs"].append(
            {"time": time.strftime("%H:%M:%S"), "message": msg}
        )
        pipeline_status["message"] = msg
    logger.info(msg)


def _update_progress(stage, progress, total, message=""):
    """Update the pipeline progress."""
    with _pipeline_lock:
        pipeline_status["stage"] = stage
        pipeline_status["progress"] = progress
        pipeline_status["total"] = total
        if message:
            pipeline_status["message"] = message


# ---------------------------------------------------------------------------
# Helper: build Config from session / request data
# ---------------------------------------------------------------------------


def _build_config(api_key=None, model_name=None, batch_size=None,
                  input_file=None, title_col=None, desc_col=None):
    """Build a Config object from provided parameters, falling back to session values."""
    overrides = {}

    model = model_name or session.get("model_name", "claude-sonnet-4-20250514")
    provider = "openai" if model.startswith("gpt-") else "anthropic"
    overrides["model"] = {"name": model, "provider": provider}

    bs = batch_size or session.get("batch_size", 25)
    overrides["input"] = {"batch_size": int(bs)}

    if input_file:
        overrides["input"]["file"] = str(input_file)
    elif session.get("input_file"):
        overrides["input"]["file"] = session["input_file"]

    if title_col or session.get("title_col"):
        overrides["input"].setdefault("columns", {})["title"] = title_col or session.get("title_col", "Title")
    if desc_col or session.get("desc_col"):
        overrides["input"].setdefault("columns", {})["description"] = desc_col or session.get("desc_col", "Description")

    overrides["output"] = {"dir": str(OUTPUT_DIR)}

    # Set the API key in the environment temporarily
    key = api_key or session.get("api_key", "")
    if key:
        if provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = key
        else:
            os.environ["OPENAI_API_KEY"] = key

    config = Config(config_path=str(PROJECT_ROOT / "config.yaml"), overrides=overrides)
    return config


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload an Excel/CSV file, detect columns, return preview."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in (".xlsx", ".xls", ".csv"):
        return jsonify({"error": f"Unsupported file type: {ext}. Use .xlsx, .xls, or .csv"}), 400

    # Save the file
    save_path = DATA_DIR / file.filename
    file.save(str(save_path))
    session["input_file"] = str(save_path)
    session["input_filename"] = file.filename

    # Read and preview
    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(save_path, engine="openpyxl")
        else:
            df = pd.read_csv(save_path)

        columns = df.columns.tolist()

        # Suggest title and description columns
        title_guess = None
        desc_guess = None
        for col in columns:
            cl = col.lower()
            if "title" in cl and title_guess is None:
                title_guess = col
            if any(kw in cl for kw in ("description", "abstract", "summary", "text", "content")) and desc_guess is None:
                desc_guess = col

        if title_guess is None and len(columns) >= 1:
            title_guess = columns[0]
        if desc_guess is None and len(columns) >= 2:
            desc_guess = columns[1]

        # Preview first 5 rows
        preview = df.head(5).fillna("").to_dict(orient="records")

        return jsonify({
            "success": True,
            "filename": file.filename,
            "total_rows": len(df),
            "columns": columns,
            "title_guess": title_guess,
            "description_guess": desc_guess,
            "preview": preview,
        })

    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 400


@app.route("/configure", methods=["POST"])
def configure():
    """Save configuration to session."""
    data = request.json or {}

    if data.get("model_name"):
        session["model_name"] = data["model_name"]
    if data.get("batch_size"):
        session["batch_size"] = int(data["batch_size"])
    if data.get("api_key"):
        session["api_key"] = data["api_key"]
    if data.get("title_col"):
        session["title_col"] = data["title_col"]
    if data.get("desc_col"):
        session["desc_col"] = data["desc_col"]

    return jsonify({"success": True, "message": "Configuration saved"})


@app.route("/estimate", methods=["GET"])
def estimate():
    """Run cost estimation without needing an API key."""
    input_file = session.get("input_file")
    if not input_file or not Path(input_file).exists():
        return jsonify({"error": "No input file uploaded. Please upload a file first."}), 400

    try:
        config = _build_config()
        loader = DataLoader(config)
        loader.load(input_file)
        stats = loader.get_stats()

        batch_size = config.batch_size
        n_batches = stats["total_batches"]

        # Compute estimates for each model
        model_estimates = []
        for model_name, pricing in MODEL_PRICING.items():
            # Stage 1: Extraction
            avg_batch_data_tokens = stats["avg_chars_per_row"] * batch_size / 4
            prompt_template_tokens = len(EXTRACTION_PROMPT) / 4
            s1_input = int((avg_batch_data_tokens + prompt_template_tokens) * n_batches)
            s1_output = int(n_batches * 2000)

            # Stage 2: Taxonomy
            estimated_concepts = int(n_batches * 15)
            s2_input = int(len(TAXONOMY_PROMPT) / 4 + estimated_concepts * 50)
            s2_output = 4000

            # Stage 3: Cross-linking
            s3_input = int(len(CROSSLINK_PROMPT) / 4 + 10000)
            s3_output = 3000

            total_input = s1_input + s2_input + s3_input
            total_output = s1_output + s2_output + s3_output
            total_cost = (total_input / 1_000_000) * pricing["input"] + (total_output / 1_000_000) * pricing["output"]

            model_estimates.append({
                "model": model_name,
                "input_price": pricing["input"],
                "output_price": pricing["output"],
                "stages": {
                    "extraction": {
                        "input_tokens": s1_input,
                        "output_tokens": s1_output,
                        "cost": round((s1_input / 1e6) * pricing["input"] + (s1_output / 1e6) * pricing["output"], 4),
                    },
                    "taxonomy": {
                        "input_tokens": s2_input,
                        "output_tokens": s2_output,
                        "cost": round((s2_input / 1e6) * pricing["input"] + (s2_output / 1e6) * pricing["output"], 4),
                    },
                    "crosslink": {
                        "input_tokens": s3_input,
                        "output_tokens": s3_output,
                        "cost": round((s3_input / 1e6) * pricing["input"] + (s3_output / 1e6) * pricing["output"], 4),
                    },
                },
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_cost": round(total_cost, 4),
                "api_calls": n_batches + 2,
            })

        # Find current model estimate
        current_model = session.get("model_name", "claude-sonnet-4-20250514")
        current_estimate = next((m for m in model_estimates if m["model"] == current_model), model_estimates[0])

        return jsonify({
            "success": True,
            "data_stats": stats,
            "current_model": current_model,
            "current_estimate": current_estimate,
            "all_models": model_estimates,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def _run_pipeline_thread(mode="full"):
    """Run the pipeline in a background thread."""
    global pipeline_status

    try:
        api_key = session.get("api_key", "") if mode != "dry-run" else ""
        input_file = session.get("input_file", "")

        if not input_file or not Path(input_file).exists():
            raise FileNotFoundError("No input file found. Please upload a file first.")

        if mode not in ("dry-run",) and not api_key:
            raise ValueError("API key is required for live pipeline runs. Please configure your API key.")

        config = _build_config(api_key=api_key)

        _log(f"Pipeline started in '{mode}' mode")
        _log(f"Model: {config.model_name} | Provider: {config.model_provider} | Batch size: {config.batch_size}")

        with _pipeline_lock:
            pipeline_status["started_at"] = time.time()

        # ----- Stage 1: Extract -----
        _update_progress("extracting", 0, 0, "Loading input data...")
        _log("Loading input data...")

        loader = DataLoader(config)
        loader.load(input_file)
        stats = loader.get_stats()
        _log(f"Loaded {stats['total_rows']} rows, {stats['total_batches']} batches")

        batches = loader.get_batches()

        _update_progress("extracting", 0, len(batches), "Starting extraction...")

        extractor = TaxonomyExtractor(config)
        extractor.loader = loader

        is_dry_run = mode == "dry-run"

        if mode in ("full", "extract", "dry-run"):
            _log(f"Stage 1: Extracting from {len(batches)} batches" + (" (DRY RUN)" if is_dry_run else ""))

            # Custom extraction with progress tracking instead of calling extract_all
            all_concepts = []
            all_entities = []
            all_relationships = []
            all_tags = []

            output_dir = Path(config.output_dir)
            if is_dry_run:
                output_dir = output_dir / "dry_run"
            output_dir.mkdir(parents=True, exist_ok=True)

            if is_dry_run:
                # In dry run, preview a few batches and save sample prompt
                n_preview = min(3, len(batches))
                for i in range(n_preview):
                    _update_progress("extracting", i + 1, len(batches),
                                     f"Previewing batch {i + 1}/{len(batches)}")
                    extractor.extract_batch(batches[i], dry_run=True)
                    _log(f"Previewed batch {i + 1}")

                sample_text = loader.format_batch_text(batches[0])
                sample_prompt = EXTRACTION_PROMPT.format(
                    batch_size=len(batches[0]),
                    batch_text=sample_text,
                )
                (output_dir / "sample_extraction_prompt.txt").write_text(sample_prompt, encoding="utf-8")

                extractions = {
                    "concepts": [], "entities": [], "relationships": [], "result_tags": [],
                    "metadata": {"dry_run": True, "total_batches": len(batches)},
                }
            else:
                for i, batch in enumerate(batches):
                    _update_progress("extracting", i, len(batches),
                                     f"Extracting batch {i + 1}/{len(batches)}")
                    try:
                        result = extractor.extract_batch(batch, dry_run=False)
                        if result:
                            all_concepts.extend(result.get("concepts", []))
                            all_entities.extend(result.get("entities", []))
                            all_relationships.extend(result.get("relationships", []))
                            all_tags.extend(result.get("result_tags", []))
                        _log(f"Batch {i + 1}/{len(batches)} complete: "
                             f"{len(result.get('concepts', []))} concepts, "
                             f"{len(result.get('entities', []))} entities" if result else f"Batch {i + 1} returned no data")
                    except Exception as e:
                        _log(f"Batch {i + 1} failed: {e}")
                    _update_progress("extracting", i + 1, len(batches))

                extractions = {
                    "concepts": all_concepts,
                    "entities": all_entities,
                    "relationships": all_relationships,
                    "result_tags": all_tags,
                    "metadata": {
                        "dry_run": False,
                        "total_batches": len(batches),
                        "total_api_calls": extractor._total_calls,
                        "total_input_tokens": extractor._total_input_tokens,
                        "total_output_tokens": extractor._total_output_tokens,
                    },
                }
                out_path = output_dir / "extractions.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(extractions, f, indent=2, ensure_ascii=False)
                _log(f"Extractions saved: {len(all_concepts)} concepts, {len(all_entities)} entities, {len(all_relationships)} relationships")

        # ----- Load existing extractions if needed -----
        if mode in ("taxonomy", "crosslink"):
            ext_path = OUTPUT_DIR / "extractions.json"
            if not ext_path.exists():
                raise FileNotFoundError("extractions.json not found. Run extraction first.")
            with open(ext_path, "r", encoding="utf-8") as f:
                extractions = json.load(f)
            _log(f"Loaded existing extractions: {len(extractions.get('concepts', []))} concepts")

        # ----- Stage 2: Taxonomy -----
        if mode in ("full", "taxonomy") and not is_dry_run:
            _update_progress("taxonomy", 0, 1, "Building taxonomy...")
            _log("Stage 2: Building taxonomy...")

            builder = TaxonomyBuilder(config)
            concepts, merge_log = builder.deduplicate_concepts(extractions.get("concepts", []))
            entities = builder.deduplicate_entities(extractions.get("entities", []))

            _log(f"Deduplicated to {len(concepts)} concepts, {len(entities)} entities")

            n_results = extractions.get("metadata", {}).get("total_batches", 0) * config.batch_size
            taxonomy = builder.build_taxonomy(
                concepts=concepts, entities=entities, n_results=n_results, dry_run=False
            )

            # Save deduplicated data
            dedup_path = OUTPUT_DIR / "deduplicated.json"
            with open(dedup_path, "w", encoding="utf-8") as f:
                json.dump({"concepts": concepts, "entities": entities, "merge_log": merge_log},
                          f, indent=2, ensure_ascii=False)

            _update_progress("taxonomy", 1, 1, "Taxonomy built")
            _log("Taxonomy construction complete")

        # ----- Stage 3: Cross-link -----
        if mode in ("full", "crosslink") and not is_dry_run:
            _update_progress("crosslinking", 0, 1, "Cross-linking...")
            _log("Stage 3: Cross-linking...")

            taxonomy_path = OUTPUT_DIR / "taxonomy.json"
            if not taxonomy_path.exists():
                raise FileNotFoundError("taxonomy.json not found. Run taxonomy building first.")

            with open(taxonomy_path, "r", encoding="utf-8") as f:
                taxonomy = json.load(f)

            dedup_path = OUTPUT_DIR / "deduplicated.json"
            if dedup_path.exists():
                with open(dedup_path, "r", encoding="utf-8") as f:
                    dedup = json.load(f)
                entities = dedup.get("entities", [])
            else:
                entities = extractions.get("entities", [])

            relationships = extractions.get("relationships", [])
            n_results = extractions.get("metadata", {}).get("total_batches", 0) * config.batch_size

            builder = TaxonomyBuilder(config)
            builder.cross_link(
                taxonomy=taxonomy, relationships=relationships, entities=entities,
                n_results=n_results, dry_run=False
            )

            _update_progress("crosslinking", 1, 1, "Cross-linking complete")
            _log("Cross-linking complete")

        # ----- Export -----
        if mode in ("full",) and not is_dry_run:
            _update_progress("exporting", 0, 1, "Exporting results...")
            _log("Exporting results in all formats...")

            _run_export(config)

            _update_progress("exporting", 1, 1, "Export complete")
            _log("All exports complete")

        # ----- Done -----
        with _pipeline_lock:
            pipeline_status["stage"] = "complete"
            pipeline_status["finished_at"] = time.time()
            elapsed = pipeline_status["finished_at"] - pipeline_status["started_at"]
            pipeline_status["message"] = f"Pipeline complete in {elapsed:.1f}s"

        _log(f"Pipeline finished successfully in {elapsed:.1f} seconds")

    except Exception as e:
        logger.exception("Pipeline failed")
        with _pipeline_lock:
            pipeline_status["stage"] = "error"
            pipeline_status["error"] = str(e)
            pipeline_status["message"] = f"Error: {str(e)}"
            pipeline_status["finished_at"] = time.time()
        _log(f"Pipeline failed: {e}")


def _run_export(config):
    """Run all exports."""
    # Load all available data
    taxonomy_tree = []
    concepts = []
    entities = []
    relationships = []
    new_rels = []

    taxonomy_path = OUTPUT_DIR / "taxonomy.json"
    extractions_path = OUTPUT_DIR / "extractions.json"
    crosslinks_path = OUTPUT_DIR / "crosslinks.json"
    dedup_path = OUTPUT_DIR / "deduplicated.json"

    if taxonomy_path.exists():
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy_data = json.load(f)
        taxonomy_tree = taxonomy_data.get("taxonomy", [])

    if extractions_path.exists():
        with open(extractions_path, "r", encoding="utf-8") as f:
            extractions = json.load(f)
        relationships = extractions.get("relationships", [])

    if crosslinks_path.exists():
        with open(crosslinks_path, "r", encoding="utf-8") as f:
            crosslinks = json.load(f)
        new_rels = crosslinks.get("new_relationships", [])

    if dedup_path.exists():
        with open(dedup_path, "r", encoding="utf-8") as f:
            dedup_data = json.load(f)
        concepts = dedup_data.get("concepts", [])
        entities = dedup_data.get("entities", [])
    elif extractions_path.exists():
        concepts = extractions.get("concepts", [])
        entities = extractions.get("entities", [])

    all_relationships = relationships + new_rels

    # Build knowledge graph
    graph = KnowledgeGraph()
    graph.add_concepts(concepts)
    graph.add_entities(entities)
    graph.add_relationships(all_relationships)
    if taxonomy_tree:
        graph.add_taxonomy_edges(taxonomy_tree)

    analysis = graph.analyze()

    # Export all formats
    graph.export_json(str(OUTPUT_DIR / "knowledge_graph.json"))
    graph.export_graphml(str(OUTPUT_DIR / "knowledge_graph.graphml"))
    graph.export_markdown(str(OUTPUT_DIR / "knowledge_graph.md"))
    graph.export_html(str(OUTPUT_DIR / "knowledge_graph.html"))

    if taxonomy_tree:
        TaxonomyExporter.export_taxonomy_markdown(taxonomy_tree, str(OUTPUT_DIR / "taxonomy_tree.md"))
    if all_relationships:
        TaxonomyExporter.export_relationships_csv(all_relationships, str(OUTPUT_DIR / "relationships.csv"))

    TaxonomyExporter.export_summary_report(
        taxonomy=taxonomy_tree, analysis=analysis,
        concepts=concepts, entities=entities, relationships=all_relationships,
        path=str(OUTPUT_DIR / "summary_report.html"),
    )
    TaxonomyExporter.export_obsidian_vault(
        taxonomy=taxonomy_tree, relationships=all_relationships, entities=entities,
        vault_path=str(OUTPUT_DIR / "obsidian_vault"),
    )

    _log(f"Exported: {analysis['total_nodes']} nodes, {analysis['total_edges']} edges")


# ---------------------------------------------------------------------------
# Pipeline run endpoints
# ---------------------------------------------------------------------------

def _start_pipeline(mode):
    """Start the pipeline in a background thread."""
    global pipeline_status

    if pipeline_status["stage"] not in ("idle", "complete", "error"):
        return jsonify({"error": "Pipeline is already running"}), 409

    _reset_status()

    # We need to copy session data since the thread won't have access to the request context
    thread_session_data = {
        "api_key": session.get("api_key", ""),
        "model_name": session.get("model_name", "claude-sonnet-4-20250514"),
        "batch_size": session.get("batch_size", 25),
        "input_file": session.get("input_file", ""),
        "title_col": session.get("title_col", "Title"),
        "desc_col": session.get("desc_col", "Description"),
    }

    def _thread_wrapper():
        # Monkey-patch session access in the thread via a simple dict
        # We store data directly and pass it through the config builder
        _run_pipeline_thread_with_data(mode, thread_session_data)

    thread = threading.Thread(target=_thread_wrapper, daemon=True)
    thread.start()

    return jsonify({"success": True, "message": f"Pipeline started in '{mode}' mode"})


def _run_pipeline_thread_with_data(mode, session_data):
    """Run pipeline with explicit session data (for use in threads)."""
    global pipeline_status

    try:
        api_key = session_data.get("api_key", "") if mode != "dry-run" else ""
        input_file = session_data.get("input_file", "")

        if not input_file or not Path(input_file).exists():
            raise FileNotFoundError("No input file found. Please upload a file first.")

        if mode not in ("dry-run",) and not api_key:
            raise ValueError("API key is required for live pipeline runs. Please configure your API key.")

        config = _build_config(
            api_key=api_key,
            model_name=session_data.get("model_name"),
            batch_size=session_data.get("batch_size"),
            input_file=input_file,
            title_col=session_data.get("title_col"),
            desc_col=session_data.get("desc_col"),
        )

        _log(f"Pipeline started in '{mode}' mode")
        _log(f"Model: {config.model_name} | Provider: {config.model_provider} | Batch size: {config.batch_size}")

        with _pipeline_lock:
            pipeline_status["started_at"] = time.time()

        is_dry_run = mode == "dry-run"
        extractions = {}

        # ----- Stage 1: Extract -----
        if mode in ("full", "extract", "dry-run"):
            _update_progress("extracting", 0, 0, "Loading input data...")
            _log("Loading input data...")

            loader = DataLoader(config)
            loader.load(input_file)
            stats = loader.get_stats()
            _log(f"Loaded {stats['total_rows']} rows, {stats['total_batches']} batches")

            batches = loader.get_batches()
            _update_progress("extracting", 0, len(batches), "Starting extraction...")

            extractor = TaxonomyExtractor(config)
            extractor.loader = loader

            all_concepts = []
            all_entities = []
            all_relationships = []
            all_tags = []

            output_dir = Path(config.output_dir)
            if is_dry_run:
                output_dir = output_dir / "dry_run"
            output_dir.mkdir(parents=True, exist_ok=True)

            if is_dry_run:
                n_preview = min(3, len(batches))
                for i in range(n_preview):
                    _update_progress("extracting", i + 1, n_preview,
                                     f"Previewing batch {i + 1}/{n_preview}")
                    extractor.extract_batch(batches[i], dry_run=True)
                    _log(f"Previewed batch {i + 1}")

                sample_text = loader.format_batch_text(batches[0])
                sample_prompt = EXTRACTION_PROMPT.format(
                    batch_size=len(batches[0]), batch_text=sample_text,
                )
                (output_dir / "sample_extraction_prompt.txt").write_text(sample_prompt, encoding="utf-8")

                extractions = {
                    "concepts": [], "entities": [], "relationships": [], "result_tags": [],
                    "metadata": {"dry_run": True, "total_batches": len(batches)},
                }
                _log("Dry run extraction complete. Sample prompts saved.")
            else:
                for i, batch in enumerate(batches):
                    _update_progress("extracting", i, len(batches),
                                     f"Extracting batch {i + 1}/{len(batches)}")
                    try:
                        result = extractor.extract_batch(batch, dry_run=False)
                        if result:
                            all_concepts.extend(result.get("concepts", []))
                            all_entities.extend(result.get("entities", []))
                            all_relationships.extend(result.get("relationships", []))
                            all_tags.extend(result.get("result_tags", []))
                            _log(f"Batch {i + 1}/{len(batches)}: "
                                 f"+{len(result.get('concepts', []))} concepts, "
                                 f"+{len(result.get('entities', []))} entities, "
                                 f"+{len(result.get('relationships', []))} relationships")
                        else:
                            _log(f"Batch {i + 1}/{len(batches)}: no data returned")
                    except Exception as e:
                        _log(f"Batch {i + 1} failed: {e}")
                    _update_progress("extracting", i + 1, len(batches))

                    # Save checkpoint every 5 batches for live preview
                    if (i + 1) % 5 == 0 or i == len(batches) - 1:
                        checkpoint = {
                            "concepts": all_concepts,
                            "entities": all_entities,
                            "relationships": all_relationships,
                            "result_tags": all_tags,
                            "checkpoint_batch": i + 1,
                        }
                        try:
                            cp_path = output_dir / "extractions_checkpoint.json"
                            with open(cp_path, "w", encoding="utf-8") as f:
                                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
                        except IOError:
                            pass

                extractions = {
                    "concepts": all_concepts,
                    "entities": all_entities,
                    "relationships": all_relationships,
                    "result_tags": all_tags,
                    "metadata": {
                        "dry_run": False,
                        "total_batches": len(batches),
                        "total_api_calls": extractor._total_calls,
                        "total_input_tokens": extractor._total_input_tokens,
                        "total_output_tokens": extractor._total_output_tokens,
                    },
                }
                out_path = output_dir / "extractions.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(extractions, f, indent=2, ensure_ascii=False)
                _log(f"Extractions saved: {len(all_concepts)} concepts, "
                     f"{len(all_entities)} entities, {len(all_relationships)} relationships")

        # ----- Load existing extractions if needed -----
        if mode in ("taxonomy", "crosslink"):
            ext_path = OUTPUT_DIR / "extractions.json"
            if not ext_path.exists():
                raise FileNotFoundError("extractions.json not found. Run extraction first.")
            with open(ext_path, "r", encoding="utf-8") as f:
                extractions = json.load(f)
            _log(f"Loaded existing extractions: {len(extractions.get('concepts', []))} concepts")

        # ----- Stage 2: Taxonomy -----
        if mode in ("full", "taxonomy") and not is_dry_run:
            _update_progress("taxonomy", 0, 1, "Building taxonomy...")
            _log("Stage 2: Building taxonomy...")

            builder = TaxonomyBuilder(config)
            concepts, merge_log = builder.deduplicate_concepts(extractions.get("concepts", []))
            entities = builder.deduplicate_entities(extractions.get("entities", []))

            _log(f"Deduplicated to {len(concepts)} concepts, {len(entities)} entities")

            n_results = extractions.get("metadata", {}).get("total_batches", 0) * config.batch_size
            builder.build_taxonomy(
                concepts=concepts, entities=entities, n_results=n_results, dry_run=False
            )

            dedup_path = OUTPUT_DIR / "deduplicated.json"
            with open(dedup_path, "w", encoding="utf-8") as f:
                json.dump({"concepts": concepts, "entities": entities, "merge_log": merge_log},
                          f, indent=2, ensure_ascii=False)

            _update_progress("taxonomy", 1, 1, "Taxonomy built")
            _log("Taxonomy construction complete")

        # ----- Stage 3: Cross-link -----
        if mode in ("full", "crosslink") and not is_dry_run:
            _update_progress("crosslinking", 0, 1, "Cross-linking...")
            _log("Stage 3: Cross-linking...")

            taxonomy_path = OUTPUT_DIR / "taxonomy.json"
            if not taxonomy_path.exists():
                raise FileNotFoundError("taxonomy.json not found. Run taxonomy building first.")

            with open(taxonomy_path, "r", encoding="utf-8") as f:
                taxonomy = json.load(f)

            dedup_path = OUTPUT_DIR / "deduplicated.json"
            if dedup_path.exists():
                with open(dedup_path, "r", encoding="utf-8") as f:
                    dedup = json.load(f)
                entities = dedup.get("entities", [])
            else:
                entities = extractions.get("entities", [])

            relationships = extractions.get("relationships", [])
            n_results = extractions.get("metadata", {}).get("total_batches", 0) * config.batch_size

            builder = TaxonomyBuilder(config)
            builder.cross_link(
                taxonomy=taxonomy, relationships=relationships, entities=entities,
                n_results=n_results, dry_run=False
            )

            _update_progress("crosslinking", 1, 1, "Cross-linking complete")
            _log("Cross-linking complete")

        # ----- Export -----
        if mode == "full" and not is_dry_run:
            _update_progress("exporting", 0, 1, "Exporting results...")
            _log("Exporting results in all formats...")
            _run_export(config)
            _update_progress("exporting", 1, 1, "Export complete")
            _log("All exports complete")

        # ----- Done -----
        with _pipeline_lock:
            pipeline_status["stage"] = "complete"
            pipeline_status["finished_at"] = time.time()
            elapsed = pipeline_status["finished_at"] - pipeline_status["started_at"]
            pipeline_status["message"] = f"Pipeline complete in {elapsed:.1f}s"

        _log(f"Pipeline finished successfully in {elapsed:.1f} seconds")

    except Exception as e:
        logger.exception("Pipeline failed")
        with _pipeline_lock:
            pipeline_status["stage"] = "error"
            pipeline_status["error"] = str(e)
            pipeline_status["message"] = f"Error: {str(e)}"
            pipeline_status["finished_at"] = time.time()
        _log(f"Pipeline failed: {e}")


@app.route("/run/extract", methods=["POST"])
def run_extract():
    return _start_pipeline("extract")


@app.route("/run/taxonomy", methods=["POST"])
def run_taxonomy():
    return _start_pipeline("taxonomy")


@app.route("/run/crosslink", methods=["POST"])
def run_crosslink():
    return _start_pipeline("crosslink")


@app.route("/run/full", methods=["POST"])
def run_full():
    return _start_pipeline("full")


@app.route("/run/dry-run", methods=["GET"])
def run_dry_run():
    return _start_pipeline("dry-run")


@app.route("/status", methods=["GET"])
def status():
    """Get current pipeline status."""
    with _pipeline_lock:
        data = {
            "stage": pipeline_status["stage"],
            "progress": pipeline_status["progress"],
            "total": pipeline_status["total"],
            "message": pipeline_status["message"],
            "error": pipeline_status["error"],
            "logs": pipeline_status["logs"][-50:],  # last 50 log entries
            "started_at": pipeline_status["started_at"],
            "finished_at": pipeline_status["finished_at"],
        }
    return jsonify(data)


# ---------------------------------------------------------------------------
# Results endpoints
# ---------------------------------------------------------------------------


@app.route("/results/live")
def results_live():
    """Get live extraction data from checkpoint file (safe to call mid-pipeline)."""
    result = {"has_data": False, "source": None}

    # First try the final extractions file
    extractions_path = OUTPUT_DIR / "extractions.json"
    checkpoint_path = OUTPUT_DIR / "extractions_checkpoint.json"

    data_path = None
    if extractions_path.exists():
        data_path = extractions_path
        result["source"] = "final"
    elif checkpoint_path.exists():
        data_path = checkpoint_path
        result["source"] = "checkpoint"

    if data_path:
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            result["has_data"] = True
            result["concepts_count"] = len(data.get("concepts", []))
            result["entities_count"] = len(data.get("entities", []))
            result["relationships_count"] = len(data.get("relationships", []))
            result["tags_count"] = len(data.get("result_tags", []))
            result["checkpoint_batch"] = data.get("checkpoint_batch", None)

            # Top 20 concepts by frequency (aggregate duplicates)
            concept_freq = {}
            for c in data.get("concepts", []):
                name = c.get("name", "").strip().lower()
                if name:
                    concept_freq[name] = concept_freq.get(name, 0) + c.get("frequency", 1)
            top_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:20]
            result["top_concepts"] = [{"name": n, "frequency": f} for n, f in top_concepts]

            # Entity type breakdown
            entity_types = {}
            for e in data.get("entities", []):
                t = e.get("type", "other")
                entity_types[t] = entity_types.get(t, 0) + 1
            result["entity_types"] = entity_types

            # Relationship type breakdown
            rel_types = {}
            for r in data.get("relationships", []):
                t = r.get("type", "related_to")
                rel_types[t] = rel_types.get(t, 0) + 1
            result["relationship_types"] = rel_types

            # Sample relationships (last 10 added)
            result["recent_relationships"] = data.get("relationships", [])[-10:]

        except (json.JSONDecodeError, IOError):
            pass  # File is being written, skip

    return jsonify(result)


@app.route("/results")
def results():
    """View results page -- redirects to main page with results tab."""
    return render_template("index.html")


@app.route("/results/graph")
def results_graph():
    """Serve the interactive HTML graph visualization."""
    graph_path = OUTPUT_DIR / "knowledge_graph.html"
    if not graph_path.exists():
        return jsonify({"error": "Graph not generated yet. Run the pipeline first."}), 404
    return send_file(str(graph_path), mimetype="text/html")


@app.route("/results/download/<fmt>")
def results_download(fmt):
    """Download exports in various formats."""
    file_map = {
        "json": ("knowledge_graph.json", "application/json"),
        "graphml": ("knowledge_graph.graphml", "application/xml"),
        "csv": ("relationships.csv", "text/csv"),
        "markdown": ("taxonomy_tree.md", "text/markdown"),
        "html": ("knowledge_graph.html", "text/html"),
        "report": ("summary_report.html", "text/html"),
        "graph_md": ("knowledge_graph.md", "text/markdown"),
        "taxonomy_json": ("taxonomy.json", "application/json"),
        "extractions": ("extractions.json", "application/json"),
        "crosslinks": ("crosslinks.json", "application/json"),
        "obsidian": (None, None),  # special: zip the vault
    }

    if fmt not in file_map:
        return jsonify({"error": f"Unknown format: {fmt}"}), 400

    if fmt == "obsidian":
        vault_dir = OUTPUT_DIR / "obsidian_vault"
        if not vault_dir.exists():
            return jsonify({"error": "Obsidian vault not generated yet."}), 404

        # Create a zip in memory
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in vault_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(vault_dir)
                    zf.write(file_path, arcname)
        buffer.seek(0)
        return send_file(buffer, mimetype="application/zip",
                         as_attachment=True, download_name="obsidian_vault.zip")

    filename, mimetype = file_map[fmt]
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": f"File not generated yet: {filename}"}), 404

    return send_file(str(file_path), mimetype=mimetype,
                     as_attachment=True, download_name=filename)


@app.route("/results/taxonomy")
def results_taxonomy():
    """Get taxonomy JSON for rendering the tree view."""
    taxonomy_path = OUTPUT_DIR / "taxonomy.json"
    if not taxonomy_path.exists():
        return jsonify({"error": "Taxonomy not generated yet."}), 404

    with open(taxonomy_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/graph-data")
def api_graph_data():
    """Return nodes and edges for D3 force graph (limited for performance)."""
    dedup_path = OUTPUT_DIR / "deduplicated.json"
    extractions_path = OUTPUT_DIR / "extractions.json"
    crosslinks_path = OUTPUT_DIR / "crosslinks.json"

    # Load concepts and entities
    concepts = []
    entities = []
    relationships = []

    if dedup_path.exists():
        with open(dedup_path, "r", encoding="utf-8") as f:
            dedup = json.load(f)
        concepts = dedup.get("concepts", [])
        entities = dedup.get("entities", [])
    elif extractions_path.exists():
        with open(extractions_path, "r", encoding="utf-8") as f:
            ext = json.load(f)
        concepts = ext.get("concepts", [])
        entities = ext.get("entities", [])

    if extractions_path.exists():
        with open(extractions_path, "r", encoding="utf-8") as f:
            ext = json.load(f)
        relationships = ext.get("relationships", [])

    # Add cross-link relationships
    if crosslinks_path.exists():
        with open(crosslinks_path, "r", encoding="utf-8") as f:
            crosslinks = json.load(f)
        relationships.extend(crosslinks.get("new_relationships", []))

    if not concepts and not entities:
        return jsonify({"error": "No data available. Run the pipeline first."}), 404

    # Build a name-based lookup for all nodes
    node_map = {}

    # Top concepts by frequency (limit to top 100)
    top_concepts = sorted(concepts, key=lambda x: x.get("frequency", 0), reverse=True)[:100]
    for c in top_concepts:
        name = c.get("name", "")
        if name and name not in node_map:
            node_map[name] = {
                "id": name,
                "label": name,
                "type": "concept",
                "category": c.get("category", ""),
                "frequency": c.get("frequency", 1),
                "description": c.get("description", ""),
            }

    # Top entities (limit to top 100)
    entity_names = set()
    for e in entities:
        entity_names.add(e.get("name", ""))
    for e in entities[:100]:
        name = e.get("name", "")
        if name and name not in node_map:
            node_map[name] = {
                "id": name,
                "label": name,
                "type": "entity",
                "category": e.get("type", "other"),
                "frequency": 1,
                "description": e.get("description", ""),
            }

    # Build edges -- only keep those that reference existing nodes
    concept_names = {c.get("name", "") for c in top_concepts}
    all_node_names = set(node_map.keys())

    # First pass: count connections for each node to find most connected
    connection_count = {}
    valid_rels = []
    for r in relationships:
        src = r.get("source", "")
        tgt = r.get("target", "")
        if src and tgt:
            connection_count[src] = connection_count.get(src, 0) + 1
            connection_count[tgt] = connection_count.get(tgt, 0) + 1
            valid_rels.append(r)

    # Add nodes that appear in relationships but are not yet in node_map
    # (only the most connected ones, up to ~200 total)
    remaining_slots = 200 - len(node_map)
    if remaining_slots > 0:
        mentioned = set()
        for r in valid_rels:
            mentioned.add(r["source"])
            mentioned.add(r["target"])
        new_names = mentioned - all_node_names
        ranked = sorted(new_names, key=lambda n: connection_count.get(n, 0), reverse=True)
        for name in ranked[:remaining_slots]:
            if name not in node_map:
                # Determine type
                ntype = "entity" if name in entity_names else "concept"
                node_map[name] = {
                    "id": name,
                    "label": name,
                    "type": ntype,
                    "category": "",
                    "frequency": 1,
                    "description": "",
                }

    all_node_names = set(node_map.keys())

    # Build final edges (only between existing nodes)
    links = []
    seen_edges = set()
    for r in valid_rels:
        src = r.get("source", "")
        tgt = r.get("target", "")
        if src in all_node_names and tgt in all_node_names:
            edge_key = f"{src}|{r.get('type', '')}|{tgt}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                links.append({
                    "source": src,
                    "target": tgt,
                    "type": r.get("type", "related_to"),
                    "confidence": r.get("confidence", ""),
                })
            if len(links) >= 500:
                break

    nodes = list(node_map.values())
    return jsonify({"nodes": nodes, "links": links})


@app.route("/api/relationship-types")
def api_relationship_types():
    """Return counts of relationship types from extractions + crosslinks."""
    extractions_path = OUTPUT_DIR / "extractions.json"
    crosslinks_path = OUTPUT_DIR / "crosslinks.json"

    relationships = []
    if extractions_path.exists():
        with open(extractions_path, "r", encoding="utf-8") as f:
            ext = json.load(f)
        relationships.extend(ext.get("relationships", []))
    if crosslinks_path.exists():
        with open(crosslinks_path, "r", encoding="utf-8") as f:
            crosslinks = json.load(f)
        relationships.extend(crosslinks.get("new_relationships", []))

    if not relationships:
        return jsonify({"error": "No relationship data available."}), 404

    type_counts = {}
    for r in relationships:
        t = r.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    return jsonify({"types": type_counts, "total": len(relationships)})


@app.route("/results/stats")
def results_stats():
    """Get extraction and graph stats."""
    result = {"has_results": False}

    # Extraction stats
    extractions_path = OUTPUT_DIR / "extractions.json"
    if extractions_path.exists():
        with open(extractions_path, "r", encoding="utf-8") as f:
            extractions = json.load(f)
        result["has_results"] = True
        result["extraction"] = {
            "concepts": len(extractions.get("concepts", [])),
            "entities": len(extractions.get("entities", [])),
            "relationships": len(extractions.get("relationships", [])),
            "tags": len(extractions.get("result_tags", [])),
            "metadata": extractions.get("metadata", {}),
        }

    # Deduplicated stats
    dedup_path = OUTPUT_DIR / "deduplicated.json"
    if dedup_path.exists():
        with open(dedup_path, "r", encoding="utf-8") as f:
            dedup = json.load(f)
        result["deduplicated"] = {
            "concepts": len(dedup.get("concepts", [])),
            "entities": len(dedup.get("entities", [])),
            "merges": len(dedup.get("merge_log", [])),
        }

    # Graph stats
    graph_json_path = OUTPUT_DIR / "knowledge_graph.json"
    if graph_json_path.exists():
        with open(graph_json_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        result["graph"] = {
            "nodes": len(graph_data.get("nodes", [])),
            "edges": len(graph_data.get("links", graph_data.get("edges", []))),
        }

    # Cross-links
    crosslinks_path = OUTPUT_DIR / "crosslinks.json"
    if crosslinks_path.exists():
        with open(crosslinks_path, "r", encoding="utf-8") as f:
            crosslinks = json.load(f)
        result["crosslinks"] = {
            "new_relationships": len(crosslinks.get("new_relationships", [])),
        }

    # Top concepts from dedup or extractions
    concepts = []
    if dedup_path.exists():
        concepts = dedup.get("concepts", [])
    elif extractions_path.exists():
        concepts = extractions.get("concepts", [])

    if concepts:
        top = sorted(concepts, key=lambda x: x.get("frequency", 0), reverse=True)[:20]
        result["top_concepts"] = top

    # Top entities
    entities = []
    if dedup_path.exists():
        entities = dedup.get("entities", [])
    elif extractions_path.exists():
        entities = extractions.get("entities", [])

    if entities:
        # Group by type
        by_type = {}
        for ent in entities:
            t = ent.get("type", "other")
            by_type.setdefault(t, []).append(ent)
        result["entities_by_type"] = {t: len(v) for t, v in by_type.items()}
        result["top_entities"] = entities[:20]

    # Available exports
    exports = []
    for name, label in [
        ("knowledge_graph.json", "JSON (Knowledge Graph)"),
        ("knowledge_graph.graphml", "GraphML (Gephi/Cytoscape)"),
        ("relationships.csv", "CSV (Relationships)"),
        ("taxonomy_tree.md", "Markdown (Taxonomy Tree)"),
        ("knowledge_graph.html", "HTML (Interactive vis.js Graph)"),
        ("summary_report.html", "HTML (Summary Report)"),
        ("knowledge_graph.md", "Markdown (Graph Summary)"),
        ("taxonomy.json", "JSON (Taxonomy Hierarchy)"),
        ("extractions.json", "JSON (Raw Extractions — Backup)"),
        ("crosslinks.json", "JSON (Cross-domain Links)"),
        ("obsidian_vault", "Obsidian Vault (Zip)"),
    ]:
        path = OUTPUT_DIR / name
        if path.exists():
            exports.append({"name": name, "label": label})
    result["available_exports"] = exports

    return jsonify(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("  \033[32m\U0001f33f Taxonomy Creator is running at http://localhost:7780\033[0m")
    print("     Open this URL in your browser to get started.")
    print()
    app.run(host="0.0.0.0", port=7780, debug=False)
