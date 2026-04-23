"""Main extraction engine for concept, entity, and relationship extraction."""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config import Config
from src.loader import DataLoader
from src.prompts import EXTRACTION_PROMPT

logger = logging.getLogger(__name__)
console = Console()


class TaxonomyExtractor:
    """Extracts concepts, entities, and relationships from text using an LLM."""

    def __init__(self, config: Config) -> None:
        """Initialize the extractor with configuration.

        Args:
            config: Application configuration object.
        """
        self.config = config
        self.client: Any = None
        self.loader = DataLoader(config)
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_calls = 0

    def _init_client(self) -> None:
        """Initialize the LLM API client based on configured provider."""
        if self.client is not None:
            return

        api_key = self.config.get_api_key()

        if self.config.model_provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            import openai
            self.client = openai.OpenAI(api_key=api_key)

    def _call_llm(self, prompt: str, dry_run: bool = False) -> Optional[str]:
        """Make an API call to the configured LLM.

        Args:
            prompt: The complete prompt to send.
            dry_run: If True, print the prompt and cost estimate without calling the API.

        Returns:
            The LLM response text, or None if dry_run is True.
        """
        estimated_input_tokens = len(prompt) // 4
        pricing = self.config.get_model_pricing()
        estimated_input_cost = (estimated_input_tokens / 1_000_000) * pricing["input"]

        if dry_run:
            console.print(Panel(
                f"[dim]{prompt[:500]}...[/dim]\n\n"
                f"[bold]Estimated input tokens:[/bold] {estimated_input_tokens:,}\n"
                f"[bold]Estimated input cost:[/bold] ${estimated_input_cost:.4f}\n"
                f"[bold]Model:[/bold] {self.config.model_name}",
                title="[yellow]DRY RUN - Prompt Preview[/yellow]",
                border_style="yellow",
            ))
            return None

        self._init_client()

        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                if self.config.model_provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.config.model_name,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    result_text = response.content[0].text
                    self._total_input_tokens += response.usage.input_tokens
                    self._total_output_tokens += response.usage.output_tokens
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    result_text = response.choices[0].message.content
                    if response.usage:
                        self._total_input_tokens += response.usage.prompt_tokens
                        self._total_output_tokens += response.usage.completion_tokens

                self._total_calls += 1
                return result_text

            except Exception as e:
                logger.warning("API call attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.info("Retrying in %.1f seconds...", wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("All %d API call attempts failed", max_retries)
                    raise

        return None  # unreachable, but satisfies type checker

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate JSON from an LLM response.

        Handles cases where the LLM wraps JSON in markdown code fences.

        Args:
            response_text: Raw text from the LLM.

        Returns:
            Parsed dictionary with concepts, entities, relationships, result_tags.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        text = response_text.strip()

        # Strip markdown code fences if present
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        # Try to find JSON object boundaries
        if not text.startswith("{"):
            brace_start = text.find("{")
            if brace_start >= 0:
                text = text[brace_start:]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            # Attempt a more aggressive extraction: find the outermost { ... }
            depth = 0
            start_idx = None
            end_idx = None
            for i, char in enumerate(text):
                if char == "{":
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        end_idx = i + 1
                        break

            if start_idx is not None and end_idx is not None:
                try:
                    data = json.loads(text[start_idx:end_idx])
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e
            else:
                raise ValueError(f"No JSON object found in LLM response: {e}") from e

        # Validate expected keys
        expected_keys = {"concepts", "entities", "relationships", "result_tags"}
        for key in expected_keys:
            if key not in data:
                data[key] = []
                logger.warning("Missing key '%s' in LLM response, defaulting to empty list", key)

        return data

    def extract_batch(self, batch: List[Dict[str, Any]], dry_run: bool = False) -> Optional[Dict[str, Any]]:
        """Extract knowledge from a single batch of research results.

        Args:
            batch: List of dicts with id, title, description.
            dry_run: If True, show the prompt without making the API call.

        Returns:
            Parsed extraction results or None if dry_run.
        """
        batch_text = self.loader.format_batch_text(batch)
        prompt = EXTRACTION_PROMPT.format(
            batch_size=len(batch),
            batch_text=batch_text,
        )

        response = self._call_llm(prompt, dry_run=dry_run)

        if response is None:
            return None

        return self._parse_response(response)

    def extract_all(
        self,
        batches: List[List[Dict[str, Any]]],
        dry_run: bool = False,
        save_intermediate: bool = True,
    ) -> Dict[str, Any]:
        """Process all batches and aggregate extraction results.

        Args:
            batches: List of batches from DataLoader.get_batches().
            dry_run: If True, show prompts and estimates without API calls.
            save_intermediate: If True, save results after each batch.

        Returns:
            Aggregated dictionary with all concepts, entities, relationships, result_tags.
        """
        all_concepts: List[Dict[str, Any]] = []
        all_entities: List[Dict[str, Any]] = []
        all_relationships: List[Dict[str, Any]] = []
        all_tags: List[Dict[str, Any]] = []

        output_dir = Path(self.config.output_dir)
        if dry_run:
            output_dir = output_dir / "dry_run"
        output_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            # In dry run mode, show estimates for a few batches
            n_preview = min(3, len(batches))
            console.print(f"\n[bold yellow]DRY RUN:[/bold yellow] Previewing {n_preview} of {len(batches)} batches\n")

            for i in range(n_preview):
                console.print(f"\n[bold]--- Batch {i + 1}/{len(batches)} ---[/bold]")
                self.extract_batch(batches[i], dry_run=True)

            # Save a full sample prompt
            sample_text = self.loader.format_batch_text(batches[0])
            sample_prompt = EXTRACTION_PROMPT.format(
                batch_size=len(batches[0]),
                batch_text=sample_text,
            )
            sample_path = output_dir / "sample_extraction_prompt.txt"
            sample_path.write_text(sample_prompt, encoding="utf-8")
            console.print(f"\n[green]Sample prompt saved to:[/green] {sample_path}")

            # Cost estimate
            self._print_cost_estimate(batches)

            return {
                "concepts": [],
                "entities": [],
                "relationships": [],
                "result_tags": [],
                "metadata": {"dry_run": True, "total_batches": len(batches)},
            }

        # Real execution with progress bar
        console.print(f"\n[bold]Extracting from {len(batches)} batches...[/bold]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=len(batches))

            for i, batch in enumerate(batches):
                try:
                    result = self.extract_batch(batch, dry_run=False)
                    if result:
                        all_concepts.extend(result.get("concepts", []))
                        all_entities.extend(result.get("entities", []))
                        all_relationships.extend(result.get("relationships", []))
                        all_tags.extend(result.get("result_tags", []))

                    if save_intermediate and (i + 1) % 10 == 0:
                        self._save_intermediate(
                            all_concepts, all_entities, all_relationships, all_tags,
                            output_dir, batch_num=i + 1,
                        )

                except Exception as e:
                    logger.error("Failed to process batch %d: %s", i + 1, e)
                    console.print(f"[red]Batch {i + 1} failed: {e}[/red]")

                progress.update(task, advance=1)

        aggregated = {
            "concepts": all_concepts,
            "entities": all_entities,
            "relationships": all_relationships,
            "result_tags": all_tags,
            "metadata": {
                "dry_run": False,
                "total_batches": len(batches),
                "total_api_calls": self._total_calls,
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
            },
        }

        # Save final results
        output_path = output_dir / "extractions.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Extractions saved to:[/green] {output_path}")

        self._print_extraction_summary(aggregated)

        return aggregated

    def _save_intermediate(
        self,
        concepts: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        tags: List[Dict[str, Any]],
        output_dir: Path,
        batch_num: int,
    ) -> None:
        """Save intermediate results to disk for resumability."""
        checkpoint = {
            "concepts": concepts,
            "entities": entities,
            "relationships": relationships,
            "result_tags": tags,
            "checkpoint_batch": batch_num,
        }
        path = output_dir / "extractions_checkpoint.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        logger.info("Checkpoint saved at batch %d to %s", batch_num, path)

    def _print_cost_estimate(self, batches: List[List[Dict[str, Any]]]) -> None:
        """Print a detailed cost estimate table."""
        total_chars = sum(
            sum(len(str(item["title"])) + len(str(item["description"])) for item in batch)
            for batch in batches
        )
        prompt_overhead = len(EXTRACTION_PROMPT) * len(batches)
        total_input_chars = total_chars + prompt_overhead
        estimated_input_tokens = total_input_chars // 4
        estimated_output_tokens = len(batches) * 2000  # ~2K tokens per batch response

        pricing = self.config.get_model_pricing()
        input_cost = (estimated_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (estimated_output_tokens / 1_000_000) * pricing["output"]

        table = Table(title="Extraction Cost Estimate", border_style="blue")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Total batches", f"{len(batches):,}")
        table.add_row("Total data characters", f"{total_chars:,}")
        table.add_row("Estimated input tokens", f"{estimated_input_tokens:,}")
        table.add_row("Estimated output tokens", f"{estimated_output_tokens:,}")
        table.add_row("Model", self.config.model_name)
        table.add_row("Input cost", f"${input_cost:.4f}")
        table.add_row("Output cost", f"${output_cost:.4f}")
        table.add_row("[bold]Total estimated cost[/bold]", f"[bold]${input_cost + output_cost:.4f}[/bold]")

        console.print()
        console.print(table)

    def _print_extraction_summary(self, aggregated: Dict[str, Any]) -> None:
        """Print a summary of extraction results."""
        table = Table(title="Extraction Results Summary", border_style="green")
        table.add_column("Category", style="bold")
        table.add_column("Count", justify="right")
        table.add_row("Concepts", f"{len(aggregated['concepts']):,}")
        table.add_row("Entities", f"{len(aggregated['entities']):,}")
        table.add_row("Relationships", f"{len(aggregated['relationships']):,}")
        table.add_row("Tagged results", f"{len(aggregated['result_tags']):,}")
        table.add_row("API calls", f"{self._total_calls:,}")
        table.add_row("Input tokens", f"{self._total_input_tokens:,}")
        table.add_row("Output tokens", f"{self._total_output_tokens:,}")

        console.print()
        console.print(table)
