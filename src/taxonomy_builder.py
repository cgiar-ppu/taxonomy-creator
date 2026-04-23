"""Taxonomy builder: deduplicates concepts and constructs hierarchical taxonomy via LLM."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from src.config import Config
from src.prompts import CROSSLINK_PROMPT, TAXONOMY_PROMPT

logger = logging.getLogger(__name__)
console = Console()


def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity ratio between two strings.

    Returns a value between 0.0 (completely different) and 1.0 (identical).
    """
    try:
        from Levenshtein import ratio
        return ratio(s1, s2)
    except ImportError:
        # Fallback: simple ratio based on common characters
        if not s1 or not s2:
            return 0.0
        max_len = max(len(s1), len(s2))
        common = sum(1 for a, b in zip(s1, s2) if a == b)
        return common / max_len


class TaxonomyBuilder:
    """Builds a hierarchical taxonomy from extracted concepts using LLM assistance."""

    def __init__(self, config: Config) -> None:
        """Initialize the taxonomy builder.

        Args:
            config: Application configuration object.
        """
        self.config = config
        self.client: Any = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _init_client(self) -> None:
        """Initialize the LLM API client."""
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
            dry_run: If True, show estimate without calling the API.

        Returns:
            Response text or None if dry_run.
        """
        estimated_input_tokens = len(prompt) // 4
        pricing = self.config.get_model_pricing()
        estimated_cost = (estimated_input_tokens / 1_000_000) * pricing["input"]

        if dry_run:
            console.print(Panel(
                f"[dim]{prompt[:800]}...[/dim]\n\n"
                f"[bold]Estimated input tokens:[/bold] {estimated_input_tokens:,}\n"
                f"[bold]Estimated input cost:[/bold] ${estimated_cost:.4f}\n"
                f"[bold]Model:[/bold] {self.config.model_name}",
                title="[yellow]DRY RUN - Taxonomy Prompt Preview[/yellow]",
                border_style="yellow",
            ))
            return None

        self._init_client()
        import time

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

                return result_text

            except Exception as e:
                logger.warning("API call attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise

        return None

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        import re

        text = text.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        if not text.startswith("{"):
            idx = text.find("{")
            if idx >= 0:
                text = text[idx:]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find outermost braces
            depth = 0
            start = None
            for i, ch in enumerate(text):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        return json.loads(text[start:i + 1])
            raise ValueError("Could not parse JSON from LLM response")

    def deduplicate_concepts(self, concepts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Deduplicate concepts using frequency counting and string similarity.

        Args:
            concepts: Raw list of concept dicts from extraction.

        Returns:
            Tuple of (deduplicated concepts, merge log entries).
        """
        threshold = self.config.merge_threshold

        # Aggregate by normalized name
        name_map: Dict[str, Dict[str, Any]] = {}
        for concept in concepts:
            name = concept.get("name", "").strip().lower()
            if not name:
                continue

            if name in name_map:
                name_map[name]["frequency"] = name_map[name].get("frequency", 1) + concept.get("frequency", 1)
            else:
                name_map[name] = {
                    "name": name,
                    "category": concept.get("category", "approach"),
                    "description": concept.get("description", ""),
                    "frequency": concept.get("frequency", 1),
                    "aliases": [],
                }

        # Now merge near-duplicates using Levenshtein distance
        names = list(name_map.keys())
        merged_into: Dict[str, str] = {}  # maps merged name -> canonical name
        merge_log: List[Dict[str, Any]] = []

        for i in range(len(names)):
            if names[i] in merged_into:
                continue
            for j in range(i + 1, len(names)):
                if names[j] in merged_into:
                    continue

                similarity = _levenshtein_ratio(names[i], names[j])
                if similarity >= threshold:
                    # Merge j into i (keep the one with higher frequency)
                    canonical = names[i]
                    to_merge = names[j]

                    if name_map[to_merge]["frequency"] > name_map[canonical]["frequency"]:
                        canonical, to_merge = to_merge, canonical

                    name_map[canonical]["frequency"] += name_map[to_merge]["frequency"]
                    name_map[canonical]["aliases"].append(to_merge)
                    name_map[canonical]["aliases"].extend(name_map[to_merge].get("aliases", []))

                    merged_into[to_merge] = canonical
                    merge_log.append({
                        "merged": [to_merge],
                        "into": canonical,
                        "reason": f"Levenshtein similarity {similarity:.2f} >= {threshold}",
                    })

        # Build deduplicated list
        deduped = [
            entry for name, entry in name_map.items()
            if name not in merged_into
        ]

        # Sort by frequency descending
        deduped.sort(key=lambda x: x.get("frequency", 0), reverse=True)

        console.print(f"[green]Deduplicated {len(concepts)} raw concepts into {len(deduped)} unique concepts[/green]")
        console.print(f"[green]Merged {len(merge_log)} near-duplicate pairs[/green]")

        return deduped, merge_log

    def deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities by name and type.

        Args:
            entities: Raw list of entity dicts from extraction.

        Returns:
            Deduplicated list of entities.
        """
        seen: Dict[str, Dict[str, Any]] = {}
        for entity in entities:
            key = (entity.get("name", "").strip().lower(), entity.get("type", ""))
            if key[0] and key not in seen:
                seen[key] = entity

        deduped = list(seen.values())
        console.print(f"[green]Deduplicated {len(entities)} raw entities into {len(deduped)} unique entities[/green]")
        return deduped

    def build_taxonomy(
        self,
        concepts: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        n_results: int,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Build a hierarchical taxonomy from deduplicated concepts.

        Args:
            concepts: Deduplicated concept list.
            entities: Deduplicated entity list (for context).
            n_results: Total number of research results processed.
            dry_run: If True, show the prompt without calling the API.

        Returns:
            Taxonomy dict or None if dry_run.
        """
        prompt = TAXONOMY_PROMPT.format(
            n_results=n_results,
            max_depth=self.config.max_depth,
            min_frequency=self.config.min_frequency,
            concepts_json=json.dumps(concepts, indent=2, ensure_ascii=False),
            entities_json=json.dumps(entities[:100], indent=2, ensure_ascii=False),  # limit for token budget
        )

        if dry_run:
            output_dir = Path(self.config.output_dir) / "dry_run"
            output_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = output_dir / "sample_taxonomy_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            console.print(f"[green]Sample taxonomy prompt saved to:[/green] {prompt_path}")

            self._call_llm(prompt, dry_run=True)
            return None

        console.print("[bold]Building taxonomy via LLM...[/bold]")
        response = self._call_llm(prompt, dry_run=False)
        if response is None:
            raise RuntimeError("LLM returned no response for taxonomy building")

        taxonomy_data = self._parse_json_response(response)

        # Save the taxonomy
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        taxonomy_path = output_dir / "taxonomy.json"
        with open(taxonomy_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy_data, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Taxonomy saved to:[/green] {taxonomy_path}")

        self._print_taxonomy_tree(taxonomy_data.get("taxonomy", []))
        return taxonomy_data

    def cross_link(
        self,
        taxonomy: Dict[str, Any],
        relationships: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        n_results: int,
        dry_run: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Find missing relationships across the taxonomy.

        Args:
            taxonomy: The built taxonomy dict.
            relationships: Existing relationships from extraction.
            entities: Deduplicated entities.
            n_results: Total number of research results.
            dry_run: If True, show the prompt without calling the API.

        Returns:
            Cross-link results or None if dry_run.
        """
        # Limit relationship list for token budget
        rel_sample = relationships[:200] if len(relationships) > 200 else relationships
        ent_sample = entities[:100] if len(entities) > 100 else entities

        prompt = CROSSLINK_PROMPT.format(
            n_results=n_results,
            taxonomy_json=json.dumps(taxonomy.get("taxonomy", []), indent=2, ensure_ascii=False),
            relationships_json=json.dumps(rel_sample, indent=2, ensure_ascii=False),
            n_relationships=len(relationships),
            entities_json=json.dumps(ent_sample, indent=2, ensure_ascii=False),
        )

        if dry_run:
            output_dir = Path(self.config.output_dir) / "dry_run"
            output_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = output_dir / "sample_crosslink_prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            console.print(f"[green]Sample cross-link prompt saved to:[/green] {prompt_path}")

            self._call_llm(prompt, dry_run=True)
            return None

        console.print("[bold]Discovering cross-links via LLM...[/bold]")
        response = self._call_llm(prompt, dry_run=False)
        if response is None:
            raise RuntimeError("LLM returned no response for cross-linking")

        crosslink_data = self._parse_json_response(response)

        # Save cross-link results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        crosslink_path = output_dir / "crosslinks.json"
        with open(crosslink_path, "w", encoding="utf-8") as f:
            json.dump(crosslink_data, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Cross-links saved to:[/green] {crosslink_path}")

        new_rels = crosslink_data.get("new_relationships", [])
        console.print(f"[green]Discovered {len(new_rels)} new relationships[/green]")

        return crosslink_data

    def _print_taxonomy_tree(self, taxonomy: List[Dict[str, Any]], max_display_depth: int = 3) -> None:
        """Print the taxonomy as a rich Tree in the console."""
        tree = Tree("[bold]Taxonomy[/bold]")

        def _add_children(parent_tree: Tree, children: List[Dict[str, Any]], depth: int) -> None:
            for child in children:
                name = child.get("name", "Unknown")
                freq = child.get("frequency", "")
                freq_str = f" ({freq})" if freq else ""
                label = f"[cyan]{name}[/cyan]{freq_str}"

                node = parent_tree.add(label)
                if depth < max_display_depth and "children" in child:
                    _add_children(node, child["children"], depth + 1)
                elif "children" in child and child["children"]:
                    node.add(f"[dim]... {len(child['children'])} more[/dim]")

        for domain in taxonomy:
            name = domain.get("name", "Unknown Domain")
            desc = domain.get("description", "")
            domain_node = tree.add(f"[bold magenta]{name}[/bold magenta] - [dim]{desc}[/dim]")
            if "children" in domain:
                _add_children(domain_node, domain["children"], 1)

        console.print()
        console.print(tree)
