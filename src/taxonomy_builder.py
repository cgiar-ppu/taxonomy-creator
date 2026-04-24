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
from src.prompts import CROSSLINK_PROMPT, TAXONOMY_DOMAIN_PROMPT, TAXONOMY_PROMPT, TAXONOMY_SKELETON_PROMPT

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

    def _call_llm(self, prompt: str, dry_run: bool = False, max_tokens_override: int | None = None) -> Optional[str]:
        """Make an API call to the configured LLM.

        Args:
            prompt: The complete prompt to send.
            dry_run: If True, show estimate without calling the API.
            max_tokens_override: Override config max_tokens for this call (useful for
                taxonomy/crosslink calls that need much larger output).

        Returns:
            Response text or None if dry_run.
        """
        max_tokens = max_tokens_override or self.config.max_tokens
        estimated_input_tokens = len(prompt) // 4
        pricing = self.config.get_model_pricing()
        estimated_cost = (estimated_input_tokens / 1_000_000) * pricing["input"]

        if dry_run:
            console.print(Panel(
                f"[dim]{prompt[:800]}...[/dim]\n\n"
                f"[bold]Estimated input tokens:[/bold] {estimated_input_tokens:,}\n"
                f"[bold]Estimated input cost:[/bold] ${estimated_cost:.4f}\n"
                f"[bold]Model:[/bold] {self.config.model_name}\n"
                f"[bold]Max output tokens:[/bold] {max_tokens:,}",
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
                        max_tokens=max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    result_text = response.content[0].text
                    self._total_input_tokens += response.usage.input_tokens
                    self._total_output_tokens += response.usage.output_tokens
                    # Log if response was truncated (stop_reason != "end_turn")
                    if response.stop_reason != "end_turn":
                        logger.warning(
                            "LLM response may be truncated (stop_reason=%s, max_tokens=%d). "
                            "Consider increasing max_tokens.",
                            response.stop_reason, max_tokens,
                        )
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        max_tokens=max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    result_text = response.choices[0].message.content
                    if response.usage:
                        self._total_input_tokens += response.usage.prompt_tokens
                        self._total_output_tokens += response.usage.completion_tokens
                    # Log if response was truncated
                    if response.choices[0].finish_reason != "stop":
                        logger.warning(
                            "LLM response may be truncated (finish_reason=%s, max_tokens=%d). "
                            "Consider increasing max_tokens.",
                            response.choices[0].finish_reason, max_tokens,
                        )

                return result_text

            except Exception as e:
                logger.warning("API call attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise

        return None

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences and truncated output.

        Strategy:
        1. Fast path: try normal JSON parsing.
        2. If that fails, attempt to repair truncated JSON by closing open
           brackets/braces and stripping trailing incomplete tokens.
        3. Only raise ValueError if even the repaired version cannot parse.
        """
        import re

        original_len = len(text)
        text = text.strip()

        # Strip markdown code fences if present
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        # Find the start of JSON
        if not text.startswith("{"):
            idx = text.find("{")
            if idx >= 0:
                text = text[idx:]

        # --- Fast path: try normal parsing ---
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # --- Try to find complete outermost braces first ---
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
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass

        # --- Repair path: fix truncated JSON ---
        logger.warning("JSON parse failed, attempting truncated JSON repair...")
        repaired = self._repair_truncated_json(text)

        try:
            result = json.loads(repaired)
            chars_trimmed = original_len - len(repaired)
            logger.warning(
                "Truncated JSON repaired successfully. "
                "Original: %d chars, repaired: %d chars, trimmed/added: %d chars.",
                original_len, len(repaired), chars_trimmed,
            )
            return result
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse JSON from LLM response even after repair: {exc}"
            ) from exc

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """Attempt to repair truncated JSON by closing open brackets and braces.

        Steps:
        1. Strip any trailing incomplete string literal (unclosed quote).
        2. Strip trailing partial tokens (incomplete key/value after last comma).
        3. Remove trailing commas before we add closing delimiters.
        4. Close all open brackets ``[`` and braces ``{`` in correct LIFO order.

        Args:
            text: The raw JSON string that failed to parse.

        Returns:
            A best-effort repaired JSON string.
        """
        # Step 1: Detect and strip trailing unclosed string
        # Walk through to track whether we are inside a string
        in_string = False
        escape_next = False
        last_quote_pos = -1
        stack: list[str] = []  # tracks open { and [

        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                if in_string:
                    in_string = False
                else:
                    in_string = True
                    last_quote_pos = i
                continue
            if not in_string:
                if ch in ('{', '['):
                    stack.append(ch)
                elif ch == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                elif ch == ']':
                    if stack and stack[-1] == '[':
                        stack.pop()

        # If we ended inside a string, strip from the last opening quote
        if in_string and last_quote_pos >= 0:
            text = text[:last_quote_pos]
            # Recompute the stack from scratch after truncation
            stack = []
            in_string = False
            escape_next = False
            for ch in text:
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch in ('{', '['):
                        stack.append(ch)
                    elif ch == '}':
                        if stack and stack[-1] == '{':
                            stack.pop()
                    elif ch == ']':
                        if stack and stack[-1] == '[':
                            stack.pop()

        # Step 2: Strip trailing whitespace/partial tokens after last complete value
        # Remove trailing commas and whitespace
        text = text.rstrip()
        while text and text[-1] in (',', ':', ' ', '\n', '\r', '\t'):
            text = text[:-1].rstrip()

        # Step 3: Close all remaining open delimiters in LIFO order
        closers = {'[': ']', '{': '}'}
        for opener in reversed(stack):
            text += closers[opener]

        return text

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
        """Build a hierarchical taxonomy from deduplicated concepts using a multi-pass approach.

        Instead of sending all concepts in a single LLM call (which causes truncation
        for large concept sets), this method works in multiple passes:

        Pass 1 (Skeleton): Send just concept names to create top-level domains and
            second-level categories. Small response, no truncation risk.
        Pass 2-N (Domain Population): For each domain, send all concept objects and
            ask the LLM to select and organize concepts belonging to that domain.
        Final (Merge): Combine all domain results and handle unplaced concepts.

        Args:
            concepts: Deduplicated concept list.
            entities: Deduplicated entity list (for context).
            n_results: Total number of research results processed.
            dry_run: If True, show the prompt without calling the API.

        Returns:
            Taxonomy dict or None if dry_run.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Dry run mode: show skeleton prompt preview only ---
        if dry_run:
            concept_names = [c.get("name", "") for c in concepts]
            skeleton_prompt = TAXONOMY_SKELETON_PROMPT.format(
                n_concepts=len(concepts),
                n_results=n_results,
                concept_names="\n".join(f"- {name}" for name in concept_names),
            )
            dry_run_dir = output_dir / "dry_run"
            dry_run_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = dry_run_dir / "sample_taxonomy_skeleton_prompt.txt"
            prompt_path.write_text(skeleton_prompt, encoding="utf-8")
            console.print(f"[green]Sample skeleton prompt saved to:[/green] {prompt_path}")
            self._call_llm(skeleton_prompt, dry_run=True)
            return None

        # ===================================================================
        # PASS 1: Build taxonomy skeleton (domains + categories, no concepts)
        # ===================================================================
        console.print("[bold]Building taxonomy via multi-pass approach...[/bold]")
        console.print(f"[dim]Total concepts to organize: {len(concepts):,}[/dim]")
        console.print()
        console.print("[bold]Pass 1:[/bold] Building taxonomy skeleton...")

        concept_names = [c.get("name", "") for c in concepts]
        skeleton_prompt = TAXONOMY_SKELETON_PROMPT.format(
            n_concepts=len(concepts),
            n_results=n_results,
            concept_names="\n".join(f"- {name}" for name in concept_names),
        )

        skeleton_response = self._call_llm(skeleton_prompt, dry_run=False)
        if skeleton_response is None:
            raise RuntimeError("LLM returned no response for taxonomy skeleton")

        # Safety: save raw skeleton response
        raw_path = output_dir / "taxonomy_skeleton_raw_response.txt"
        raw_path.write_text(skeleton_response, encoding="utf-8")
        logger.info("Raw skeleton response saved to %s (%d chars)", raw_path, len(skeleton_response))

        skeleton_data = self._parse_json_response(skeleton_response)

        # Save skeleton
        skeleton_path = output_dir / "taxonomy_skeleton.json"
        with open(skeleton_path, "w", encoding="utf-8") as f:
            json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Skeleton saved to:[/green] {skeleton_path}")

        domains = skeleton_data.get("taxonomy", [])
        console.print(f"[green]Created {len(domains)} top-level domains[/green]")
        for domain in domains:
            n_children = len(domain.get("children", []))
            console.print(f"  [cyan]{domain.get('name', '?')}[/cyan] ({n_children} categories)")
        console.print()

        # ===================================================================
        # PASS 2-N: Populate each domain with concepts
        # ===================================================================
        taxonomy_max_tokens = max(self.config.max_tokens, 16384)
        all_concept_names_set = {c.get("name", "").strip().lower() for c in concepts}
        placed_concepts: set = set()
        populated_domains: List[Dict[str, Any]] = []

        for idx, domain in enumerate(domains, start=1):
            domain_name = domain.get("name", "Unknown")
            domain_desc = domain.get("description", "")
            subcategories = domain.get("children", [])

            console.print(f"[bold]Pass {idx + 1}:[/bold] Populating domain {idx}/{len(domains)}: [cyan]{domain_name}[/cyan]...")

            # Format subcategories for the prompt
            subcategories_lines = []
            for sub in subcategories:
                sub_name = sub.get("name", "")
                sub_desc = sub.get("description", "")
                subcategories_lines.append(f"  - {sub_name}: {sub_desc}")
            subcategories_text = "\n".join(subcategories_lines) if subcategories_lines else "  (none specified)"

            # Send only concept names (with frequency hint) instead of full JSON objects
            concept_name_lines = "\n".join(
                f"- {c.get('name', '')} (freq: {c.get('frequency', 1)})"
                for c in concepts
            )
            domain_prompt = TAXONOMY_DOMAIN_PROMPT.format(
                n_results=n_results,
                domain_name=domain_name,
                domain_description=domain_desc,
                subcategories_text=subcategories_text,
                concept_names=concept_name_lines,
                max_depth=self.config.max_depth,
                min_frequency=self.config.min_frequency,
                n_concepts=len(concepts),
            )

            domain_response = self._call_llm(
                domain_prompt, dry_run=False, max_tokens_override=taxonomy_max_tokens,
            )
            if domain_response is None:
                logger.warning("LLM returned no response for domain '%s', skipping", domain_name)
                continue

            # Safety: save raw domain response
            safe_name = domain_name.lower().replace(" ", "_").replace("/", "_")
            raw_domain_path = output_dir / f"taxonomy_domain_{safe_name}_raw_response.txt"
            raw_domain_path.write_text(domain_response, encoding="utf-8")
            logger.info("Raw domain response saved to %s (%d chars)", raw_domain_path, len(domain_response))

            domain_data = self._parse_json_response(domain_response)

            # Enrich leaf nodes with full concept metadata (description, aliases, frequency)
            self._enrich_taxonomy_leaves(domain_data, concepts)

            # Track which concepts were placed in this domain
            domain_placed = domain_data.get("concepts_placed", [])
            domain_placed_lower = {name.strip().lower() for name in domain_placed}
            placed_concepts.update(domain_placed_lower)

            # Remove the concepts_placed key from the output (it was just for tracking)
            domain_data.pop("concepts_placed", None)

            populated_domains.append(domain_data)
            console.print(f"  [green]Placed {len(domain_placed_lower)} concepts[/green]")

        # ===================================================================
        # FINAL: Handle unplaced concepts and merge
        # ===================================================================
        unplaced_names = all_concept_names_set - placed_concepts
        # Remove empty strings
        unplaced_names.discard("")

        if unplaced_names:
            console.print(f"\n[yellow]{len(unplaced_names)} concepts were not placed in any domain[/yellow]")

            # Find or create the "Other / Cross-cutting" domain
            other_domain = None
            for dom in populated_domains:
                if "other" in dom.get("name", "").lower() or "cross-cutting" in dom.get("name", "").lower():
                    other_domain = dom
                    break

            # Build leaf nodes for unplaced concepts
            unplaced_leaves = []
            for concept in concepts:
                if concept.get("name", "").strip().lower() in unplaced_names:
                    unplaced_leaves.append({
                        "name": concept.get("name", ""),
                        "description": concept.get("description", ""),
                        "aliases": concept.get("aliases", []),
                        "frequency": concept.get("frequency", 1),
                    })

            if other_domain is not None:
                # Append unplaced concepts under a "Uncategorized" sub-branch
                if "children" not in other_domain:
                    other_domain["children"] = []
                other_domain["children"].append({
                    "name": "Uncategorized",
                    "description": "Concepts not assigned to a specific domain during multi-pass taxonomy building",
                    "children": unplaced_leaves,
                })
            else:
                # Create the "Other / Cross-cutting" domain from scratch
                populated_domains.append({
                    "name": "Other / Cross-cutting",
                    "description": "Concepts that span multiple domains or do not fit neatly into other categories",
                    "children": [{
                        "name": "Uncategorized",
                        "description": "Concepts not assigned to a specific domain during multi-pass taxonomy building",
                        "children": unplaced_leaves,
                    }],
                })

            console.print(f"[green]Added {len(unplaced_leaves)} unplaced concepts to 'Other / Cross-cutting' domain[/green]")

        # Build final taxonomy
        taxonomy_data = {"taxonomy": populated_domains}

        # Save the merged taxonomy
        taxonomy_path = output_dir / "taxonomy.json"
        with open(taxonomy_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy_data, f, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Taxonomy saved to:[/green] {taxonomy_path}")

        # Print summary table
        self._print_build_summary(populated_domains, len(concepts), placed_concepts, unplaced_names)

        # Print tree visualization
        self._print_taxonomy_tree(taxonomy_data.get("taxonomy", []))
        return taxonomy_data

    def _enrich_taxonomy_leaves(
        self,
        node: Dict[str, Any],
        concepts: List[Dict[str, Any]],
    ) -> int:
        """Walk the taxonomy tree and enrich leaf nodes with full concept metadata.

        For each leaf node that has a ``name`` but is missing ``description``,
        ``aliases``, or ``frequency``, look up the matching concept from the
        original concepts list and fill in the details.

        Args:
            node: A taxonomy node (may have ``children``).
            concepts: The full deduplicated concept list to look up from.

        Returns:
            The number of leaf nodes that were enriched.
        """
        # Build a lookup dict if not provided as a pre-built dict
        if isinstance(concepts, dict):
            # Already a lookup dict
            lookup = concepts
        else:
            # Build from list
            lookup = {}
            for c in concepts:
                if isinstance(c, dict):
                    key = c.get("name", "").strip().lower()
                    if key:
                        lookup[key] = c

        enriched_count = 0
        children = node.get("children", [])

        if not children:
            # This is a leaf node -- try to enrich it
            name_key = node.get("name", "").strip().lower()
            if name_key and name_key in lookup:
                source = lookup[name_key]
                if "description" not in node or not node["description"]:
                    node["description"] = source.get("description", "")
                if "aliases" not in node or not node["aliases"]:
                    node["aliases"] = source.get("aliases", [])
                if "frequency" not in node:
                    node["frequency"] = source.get("frequency", 1)
                enriched_count = 1
        else:
            for child in children:
                if isinstance(child, dict):
                    enriched_count += self._enrich_taxonomy_leaves(child, lookup)
                # Skip non-dict children (e.g. strings from truncated responses)

        return enriched_count

    def _print_build_summary(
        self,
        domains: List[Dict[str, Any]],
        total_concepts: int,
        placed: set,
        unplaced: set,
    ) -> None:
        """Print a summary table of concepts placed per domain vs total.

        Args:
            domains: The populated domain list.
            total_concepts: Total number of input concepts.
            placed: Set of placed concept names (lowercased).
            unplaced: Set of unplaced concept names (lowercased).
        """
        table = Table(title="Multi-Pass Taxonomy Build Summary")
        table.add_column("Domain", style="cyan")
        table.add_column("Concepts Placed", justify="right", style="green")

        def _count_leaves(node: Dict[str, Any]) -> int:
            """Count leaf nodes (nodes with frequency) recursively."""
            children = node.get("children", [])
            if not children:
                return 1 if "frequency" in node else 0
            return sum(_count_leaves(c) for c in children)

        total_placed_counted = 0
        for domain in domains:
            count = _count_leaves(domain)
            total_placed_counted += count
            table.add_row(domain.get("name", "?"), str(count))

        table.add_section()
        table.add_row("[bold]Total placed[/bold]", f"[bold]{total_placed_counted}[/bold]")
        table.add_row("[bold]Total input concepts[/bold]", f"[bold]{total_concepts}[/bold]")
        coverage = (len(placed) / total_concepts * 100) if total_concepts else 0
        table.add_row("[bold]Coverage[/bold]", f"[bold]{coverage:.1f}%[/bold]")
        if unplaced:
            table.add_row("[yellow]Unplaced (added to Other)[/yellow]", f"[yellow]{len(unplaced)}[/yellow]")

        console.print()
        console.print(table)

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
        # Use a generous max_tokens for cross-link discovery
        crosslink_max_tokens = max(self.config.max_tokens, 16384)
        console.print(f"[dim]Using max_tokens={crosslink_max_tokens:,} for cross-link response[/dim]")
        response = self._call_llm(prompt, dry_run=False, max_tokens_override=crosslink_max_tokens)
        if response is None:
            raise RuntimeError("LLM returned no response for cross-linking")

        # Safety: save raw response before parsing
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / "crosslink_raw_response.txt"
        raw_path.write_text(response, encoding="utf-8")
        logger.info("Raw cross-link LLM response saved to %s (%d chars)", raw_path, len(response))
        console.print(f"[dim]Raw LLM response saved to:[/dim] {raw_path}")

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
