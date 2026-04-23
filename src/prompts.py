"""Prompt templates for each extraction and taxonomy-building stage.

These prompts are the core of the taxonomy creation pipeline. They instruct the
LLM to extract structured knowledge from unstructured research result descriptions.
"""

# ---------------------------------------------------------------------------
# STAGE 1: Concept and Entity Extraction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a knowledge extraction specialist. Analyze the following {batch_size} research result descriptions and extract structured knowledge.

For each batch, identify:

## CONCEPTS (abstract ideas, methodologies, approaches, phenomena)
Extract concepts that represent reusable knowledge -- things that could appear across multiple results.
Examples: "marker-assisted selection", "climate-smart agriculture", "food security", "crop diversification"

Focus on:
- Research methodologies and approaches
- Agricultural practices and systems
- Development outcomes and goals
- Scientific phenomena and processes
- Policy frameworks and strategies

## ENTITIES (concrete, named things)
- **Organisms**: crop species, varieties, pathogens, pests (e.g., "wheat", "NERICA rice", "stem rust")
- **Organizations**: research centers, NGOs, government bodies (e.g., "CIMMYT", "IRRI", "FAO")
- **Places**: countries, regions, agroecological zones (e.g., "Sub-Saharan Africa", "South Asia", "semi-arid tropics")
- **Tools/Technologies**: specific platforms, databases, models (e.g., "DSSAT model", "SeedTracker", "drone-based phenotyping")
- **People roles**: (not names -- roles like "smallholder farmer", "extension agent", "plant breeder")

## RELATIONSHIPS
For each relationship found, specify:
- source: the concept or entity
- relationship_type: one of [is_a, part_of, uses, produces, targets, located_in, collaborates_with, addresses, related_to]
- target: the other concept or entity
- confidence: extracted (directly stated) | inferred (synthesized from context) | ambiguous (uncertain)
- evidence: brief quote or paraphrase from source text

Guidelines for relationships:
- "is_a" = taxonomic classification (e.g., "wheat is_a cereal crop")
- "part_of" = composition or membership (e.g., "gene is part_of genome")
- "uses" = methodology or tool usage (e.g., "breeding program uses marker-assisted selection")
- "produces" = output or result (e.g., "project produces improved variety")
- "targets" = intended beneficiary or focus (e.g., "innovation targets smallholder farmers")
- "located_in" = geographic context (e.g., "trial located_in Kenya")
- "collaborates_with" = partnership (e.g., "CIMMYT collaborates_with national programs")
- "addresses" = problem being solved (e.g., "intervention addresses food insecurity")
- "related_to" = general association when no specific type fits

## TAGS
Assign 2-5 domain tags to each result from an EMERGENT vocabulary (do not use a fixed list -- let tags emerge from content). Tags should be concise (1-3 words) and capture the primary domains of each result.

---

RESEARCH RESULTS TO ANALYZE:

{batch_text}

---

Respond in this exact JSON format (no markdown code fences, just raw JSON):
{{
  "concepts": [
    {{"name": "concept name in lowercase", "category": "methodology|phenomenon|outcome|approach|system", "description": "one-line definition", "frequency": 1}}
  ],
  "entities": [
    {{"name": "entity name", "type": "organism|organization|place|tool|role", "description": "one-line definition"}}
  ],
  "relationships": [
    {{"source": "source concept or entity", "type": "is_a|part_of|uses|produces|targets|located_in|collaborates_with|addresses|related_to", "target": "target concept or entity", "confidence": "extracted|inferred|ambiguous", "evidence": "brief supporting text"}}
  ],
  "result_tags": [
    {{"result_id": 1, "tags": ["tag1", "tag2"]}}
  ]
}}

Important:
- Use lowercase for concept names to enable deduplication
- Use proper capitalization for entity names (organizations, places)
- Include at least one relationship for every concept and entity when possible
- Be specific rather than vague -- "marker-assisted backcrossing" is better than "breeding"
- Every concept must have a clear, non-circular description"""

# ---------------------------------------------------------------------------
# STAGE 2: Taxonomy Construction
# ---------------------------------------------------------------------------

TAXONOMY_PROMPT = """You are a taxonomy architect. Given the following extracted concepts and entities from {n_results} research results, construct a hierarchical taxonomy.

Rules:
1. Group concepts into a multi-level hierarchy (max {max_depth} levels deep)
2. The TOP level should be broad domains. Recommended top-level categories include (but adapt based on content):
   - Crop Improvement (breeding, genetics, genomics)
   - Natural Resource Management (soil, water, land)
   - Climate Adaptation (resilience, mitigation, climate-smart approaches)
   - Food & Nutrition Security (diets, nutrition, food systems)
   - Policy & Institutions (governance, markets, trade)
   - Knowledge & Capacity (training, extension, information systems)
   - Sustainable Intensification (productivity, efficiency, inputs)
   - Social Inclusion (gender, youth, equity)
3. Each subsequent level gets more specific
4. Every concept must appear exactly once in the taxonomy
5. If a concept could belong to multiple branches, place it in the PRIMARY branch and note alternatives in the description
6. Merge near-duplicates (e.g., "marker assisted selection" and "MAS" -> single entry with aliases)
7. Concepts appearing fewer than {min_frequency} times should still be included but can be grouped under a "Miscellaneous" sub-branch of their domain
8. Preserve the frequency count from the extraction stage

EXTRACTED CONCEPTS (with frequencies):
{concepts_json}

EXTRACTED ENTITIES (for context -- do not include in taxonomy tree, but reference them):
{entities_json}

Respond in this exact JSON format (no markdown code fences, just raw JSON):
{{
  "taxonomy": [
    {{
      "name": "Top-Level Domain",
      "description": "Brief domain description",
      "children": [
        {{
          "name": "Sub-category",
          "description": "Brief sub-category description",
          "children": [
            {{
              "name": "Specific Concept",
              "description": "Brief concept description",
              "aliases": ["alternative name 1", "abbreviation"],
              "frequency": 5
            }}
          ]
        }}
      ]
    }}
  ],
  "merge_log": [
    {{"merged": ["concept A", "concept B"], "into": "canonical name", "reason": "Identical meaning, different phrasing"}}
  ],
  "unmapped_concepts": ["any concepts that genuinely do not fit the taxonomy"]
}}

Important:
- Ensure the taxonomy is MECE (Mutually Exclusive, Collectively Exhaustive) at each level
- Prefer descriptive names over acronyms at higher levels
- The taxonomy should feel natural to an agricultural researcher browsing it
- Do not create levels with only one child -- merge them up"""

# ---------------------------------------------------------------------------
# STAGE 3: Cross-Linking and Relationship Enrichment
# ---------------------------------------------------------------------------

CROSSLINK_PROMPT = """You are a knowledge graph analyst. Given the taxonomy and relationships already extracted from {n_results} agricultural research results, find MISSING connections that are strongly implied but not explicitly stated.

Look for these specific patterns:

1. **Implicit is_a relationships**: Concepts in the taxonomy that should be children of other concepts not yet linked
2. **Cross-domain bridges**: Concepts from different top-level domains that connect
   - Example: "drought tolerance" (Crop Improvement) <-> "climate adaptation" (Climate Adaptation)
   - Example: "nutrition-sensitive agriculture" bridges Food Security and Crop Improvement
3. **Entity-concept links**: Organizations that work on specific concepts
   - Example: "CIMMYT" -> "wheat improvement", "IRRI" -> "rice breeding"
4. **Methodology-outcome chains**: Approaches that lead to specific outcomes
   - Example: "participatory plant breeding" -> produces -> "farmer-preferred varieties"
5. **Geographic patterns**: Concepts or entities strongly associated with specific regions
6. **Complementary technologies**: Tools and methods that are commonly used together

CURRENT TAXONOMY:
{taxonomy_json}

CURRENT RELATIONSHIPS ({n_relationships} total):
{relationships_json}

ENTITIES FOR REFERENCE:
{entities_json}

Find 20-50 additional relationships that are strongly implied by the data. Prioritize:
- High-confidence inferences over speculative ones
- Cross-domain bridges (these are most valuable for knowledge discovery)
- Entity-concept links (these ground the taxonomy in real-world actors)

Respond in this exact JSON format (no markdown code fences, just raw JSON):
{{
  "new_relationships": [
    {{
      "source": "source concept or entity",
      "type": "is_a|part_of|uses|produces|targets|located_in|collaborates_with|addresses|related_to",
      "target": "target concept or entity",
      "confidence": "inferred",
      "evidence": "reasoning for why this relationship exists",
      "cross_domain": true
    }}
  ],
  "suggested_taxonomy_moves": [
    {{
      "concept": "concept name",
      "from_branch": "current parent",
      "to_branch": "suggested parent",
      "reason": "why this placement is better"
    }}
  ]
}}"""
