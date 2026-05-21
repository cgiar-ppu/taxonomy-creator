#!/usr/bin/env python3
"""
build_static.py -- Generate a deployable static site in dist/

Reads the Flask templates and output data files, then produces a
self-contained static site that can be served from any static file
server (e.g. AWS Amplify, Netlify, GitHub Pages).

Usage:
    python build_static.py
"""

import json
import os
import re
import shutil
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMPLATE_PATH = PROJECT_ROOT / "templates" / "index.html"
DIST_DIR = PROJECT_ROOT / "dist"
DATA_DIR = DIST_DIR / "data"
DOWNLOADS_DIR = DIST_DIR / "downloads"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"  [warn] Could not load {path.name}: {exc}")
        return {}


def save_json(data, path: Path):
    """Write data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  -> {path.relative_to(DIST_DIR)}")


# ---------------------------------------------------------------------------
# Pre-compute API-equivalent JSON payloads
# ---------------------------------------------------------------------------


def compute_graph_data(
    concepts: list, entities: list, relationships: list, node_limit: int = 500
) -> dict:
    """Replicate the /api/graph-data endpoint logic with configurable limit."""
    node_map = {}

    concept_limit = max(node_limit // 2, 50)
    entity_limit = max(node_limit // 2, 50)

    # Top concepts by frequency
    top_concepts = sorted(
        concepts, key=lambda x: x.get("frequency", 0), reverse=True
    )[:concept_limit]
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

    entity_names = {e.get("name", "") for e in entities}
    for e in entities[:entity_limit]:
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

    concept_names = {c.get("name", "") for c in top_concepts}
    all_node_names = set(node_map.keys())

    connection_count = {}
    valid_rels = []
    for r in relationships:
        src = r.get("source", "")
        tgt = r.get("target", "")
        if src and tgt:
            connection_count[src] = connection_count.get(src, 0) + 1
            connection_count[tgt] = connection_count.get(tgt, 0) + 1
            valid_rels.append(r)

    remaining_slots = node_limit - len(node_map)
    if remaining_slots > 0:
        mentioned = set()
        for r in valid_rels:
            mentioned.add(r["source"])
            mentioned.add(r["target"])
        new_names = mentioned - all_node_names
        ranked = sorted(
            new_names, key=lambda n: connection_count.get(n, 0), reverse=True
        )
        for name in ranked[:remaining_slots]:
            if name not in node_map:
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

    links = []
    seen_edges = set()
    for r in valid_rels:
        src = r.get("source", "")
        tgt = r.get("target", "")
        if src in all_node_names and tgt in all_node_names:
            edge_key = f"{src}|{r.get('type', '')}|{tgt}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                links.append(
                    {
                        "source": src,
                        "target": tgt,
                        "type": r.get("type", "related_to"),
                        "confidence": r.get("confidence", ""),
                    }
                )
            if len(links) >= 500:
                break

    return {"nodes": list(node_map.values()), "links": links}


def compute_relationship_types(relationships: list) -> dict:
    """Replicate the /api/relationship-types endpoint."""
    type_counts = {}
    for r in relationships:
        t = r.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    return {"types": type_counts, "total": len(relationships)}


def compute_concept_data_points(extractions: dict) -> dict:
    """Replicate the /api/concept-data-points endpoint.

    Reads result_tags from extractions and builds a reverse mapping:
    tag_name.lower() -> sorted list of result IDs that carry that tag.
    """
    result_tags = extractions.get("result_tags", [])
    tag_to_results: dict[str, set] = {}
    for entry in result_tags:
        rid = entry.get("result_id", "")
        tags = entry.get("tags", [])
        for tag in tags:
            key = tag.strip().lower()
            if key:
                tag_to_results.setdefault(key, set()).add(str(rid))

    # Convert sets to sorted lists for stable JSON output
    return {k: sorted(v) for k, v in tag_to_results.items()}


def compute_stats(
    extractions: dict,
    dedup: dict,
    graph_json: dict,
    crosslinks: dict,
) -> dict:
    """Replicate the /results/stats endpoint."""
    result = {"has_results": False}

    if extractions:
        result["has_results"] = True
        result["extraction"] = {
            "concepts": len(extractions.get("concepts", [])),
            "entities": len(extractions.get("entities", [])),
            "relationships": len(extractions.get("relationships", [])),
            "tags": len(extractions.get("result_tags", [])),
            "metadata": extractions.get("metadata", {}),
        }

    if dedup:
        result["deduplicated"] = {
            "concepts": len(dedup.get("concepts", [])),
            "entities": len(dedup.get("entities", [])),
            "merges": len(dedup.get("merge_log", [])),
        }

    if graph_json:
        result["graph"] = {
            "nodes": len(graph_json.get("nodes", [])),
            "edges": len(
                graph_json.get("links", graph_json.get("edges", []))
            ),
        }

    if crosslinks:
        result["crosslinks"] = {
            "new_relationships": len(
                crosslinks.get("new_relationships", [])
            ),
        }

    # Top concepts
    concepts = dedup.get("concepts", []) if dedup else extractions.get("concepts", [])
    if concepts:
        top = sorted(
            concepts, key=lambda x: x.get("frequency", 0), reverse=True
        )[:20]
        result["top_concepts"] = top

    # Entities by type
    entities = dedup.get("entities", []) if dedup else extractions.get("entities", [])
    if entities:
        by_type = {}
        for ent in entities:
            t = ent.get("type", "other")
            by_type.setdefault(t, []).append(ent)
        result["entities_by_type"] = {t: len(v) for t, v in by_type.items()}
        result["top_entities"] = entities[:20]

    # Available exports -- we will compute this from the downloads dir later
    result["available_exports"] = []

    return result


# ---------------------------------------------------------------------------
# HTML transformation
# ---------------------------------------------------------------------------

DOWNLOAD_FILE_MAP = {
    "json": "knowledge_graph.json",
    "graphml": "knowledge_graph.graphml",
    "csv": "relationships.csv",
    "markdown": "taxonomy_tree.md",
    "html": "knowledge_graph.html",
    "report": "summary_report.html",
    "graph_md": "knowledge_graph.md",
    "taxonomy_json": "taxonomy.json",
    "extractions": "extractions.json",
    "crosslinks": "crosslinks.json",
}

EXPORT_LABELS = [
    ("knowledge_graph.json", "JSON (Knowledge Graph)"),
    ("knowledge_graph.graphml", "GraphML (Gephi/Cytoscape)"),
    ("relationships.csv", "CSV (Relationships)"),
    ("taxonomy_tree.md", "Markdown (Taxonomy Tree)"),
    ("knowledge_graph.html", "HTML (Interactive vis.js Graph)"),
    ("summary_report.html", "HTML (Summary Report)"),
    ("knowledge_graph.md", "Markdown (Graph Summary)"),
    ("taxonomy.json", "JSON (Taxonomy Hierarchy)"),
    ("extractions.json", "JSON (Raw Extractions -- Backup)"),
    ("crosslinks.json", "JSON (Cross-domain Links)"),
    ("obsidian_vault.zip", "Obsidian Vault (Zip)"),
]


def transform_html(html: str) -> str:
    """Transform the Flask template into a static page."""

    # ---------------------------------------------------------------
    # 1. Replace <title>
    # ---------------------------------------------------------------
    html = html.replace(
        "<title>Taxonomy Creator</title>",
        "<title>PRMS Taxonomy Explorer -- CGIAR Research Results</title>",
    )

    # ---------------------------------------------------------------
    # 2. Replace the header content
    # ---------------------------------------------------------------
    html = html.replace(
        '<h1><span>Taxonomy</span> Creator</h1>\n'
        '        <p>Automated taxonomy extraction from unstructured text</p>',
        '<h1><span>PRMS Taxonomy Explorer</span></h1>\n'
        '        <p>CGIAR Research Results</p>',
    )

    # ---------------------------------------------------------------
    # 3. Remove the Upload, Estimate, and Run Pipeline sidebar items
    # ---------------------------------------------------------------
    # Remove Upload nav item
    html = re.sub(
        r'<div class="nav-item active" data-tab="upload"[^>]*>.*?</div>\s*',
        "",
        html,
        flags=re.DOTALL,
    )
    # Remove Estimate nav item
    html = re.sub(
        r'<div class="nav-item" data-tab="estimate"[^>]*>.*?</div>\s*',
        "",
        html,
        flags=re.DOTALL,
    )
    # Remove Run Pipeline nav item
    html = re.sub(
        r'<div class="nav-item" data-tab="run"[^>]*>.*?</div>\s*',
        "",
        html,
        flags=re.DOTALL,
    )

    # Make Results nav item the default active
    html = html.replace(
        '<div class="nav-item" data-tab="results" onclick="switchTab(\'results\')">',
        '<div class="nav-item active" data-tab="results" onclick="switchTab(\'results\')">',
    )

    # ---------------------------------------------------------------
    # 4. Remove tab content for upload, estimate, run
    # ---------------------------------------------------------------
    # Remove tab-upload content block
    html = re.sub(
        r'<!-- Tab 1: Upload & Configure -->.*?<!-- Tab 2: Estimate -->',
        "<!-- Tab 2: Estimate -->",
        html,
        flags=re.DOTALL,
    )
    # Remove tab-estimate content block
    html = re.sub(
        r'<!-- Tab 2: Estimate -->.*?<!-- Tab 3: Run Pipeline -->',
        "<!-- Tab 3: Run Pipeline -->",
        html,
        flags=re.DOTALL,
    )
    # Remove tab-run content block
    html = re.sub(
        r'<!-- Tab 3: Run Pipeline -->.*?<!-- Tab 4: Results -->',
        "<!-- Tab 4: Results -->",
        html,
        flags=re.DOTALL,
    )

    # ---------------------------------------------------------------
    # 5. Make Results tab content active by default
    # ---------------------------------------------------------------
    html = html.replace(
        '<div id="tab-results" class="tab-content">',
        '<div id="tab-results" class="tab-content active">',
    )

    # ---------------------------------------------------------------
    # 6. Remove the status badge from header (no pipeline to track)
    # ---------------------------------------------------------------
    html = re.sub(
        r'<div id="statusBadge".*?</div>\s*',
        "",
        html,
        flags=re.DOTALL,
    )

    # ---------------------------------------------------------------
    # 7. Replace all the JavaScript with static-site-friendly version
    # ---------------------------------------------------------------
    # Remove entire <script>...</script> block and replace
    html = re.sub(
        r"<script>\s*// ====+\s*// State.*?</script>",
        _build_static_js(),
        html,
        flags=re.DOTALL,
    )

    # ---------------------------------------------------------------
    # 8. Add a green CGIAR banner below the header
    # ---------------------------------------------------------------
    banner_html = """
<div style="background:linear-gradient(90deg, #00843D, #00A651); padding:8px 32px; text-align:center; font-size:13px; color:white; font-weight:600; letter-spacing:0.5px;">
    PRMS Taxonomy Explorer -- CGIAR Research Results
</div>
"""
    html = html.replace("<!-- Layout -->", banner_html + "\n<!-- Layout -->")

    return html


def _build_static_js() -> str:
    """Generate the replacement <script> block for the static site.

    This includes:
    - Tab navigation (results + visualizations only)
    - Toast notifications
    - Results loading with concept-data-points support
    - Taxonomy tree with data points panel for leaf nodes
    - Improved sunburst with HSL coloring, nav bar, segment counter
    - Improved network graph with click-to-stick, multi-hop, detail panel
    - All other chart visualizations
    All fetch URLs point to static JSON files in data/.
    """
    return """<script>
// ======================================================================
// Static Site -- Tab Navigation
// ======================================================================
var currentTab = 'results';

function switchTab(tab) {
    currentTab = tab;
    document.querySelectorAll('.tab-content').forEach(function(el) { el.classList.remove('active'); });
    document.querySelectorAll('.nav-item').forEach(function(el) { el.classList.remove('active'); });
    document.getElementById('tab-' + tab).classList.add('active');
    document.querySelector('.nav-item[data-tab="' + tab + '"]').classList.add('active');

    if (tab === 'results') loadResults();
    if (tab === 'visualizations') loadVisualizations();
}

// ======================================================================
// Toast Notifications
// ======================================================================
function showToast(message, type) {
    type = type || 'info';
    var container = document.getElementById('toastContainer');
    var toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(function() {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(function() { toast.remove(); }, 300);
    }, 4000);
}

// ======================================================================
// Utility
// ======================================================================
function escapeHtml(str) {
    if (!str) return '';
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ======================================================================
// Results
// ======================================================================
function loadResults() {
    fetch('data/stats.json')
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (!data.has_results) {
            document.getElementById('noResults').classList.remove('hidden');
            document.getElementById('resultsContent').classList.add('hidden');
            return;
        }

        document.getElementById('noResults').classList.add('hidden');
        document.getElementById('resultsContent').classList.remove('hidden');
        renderResultsStats(data);
        renderTopConcepts(data.top_concepts || []);
        renderEntitiesTable(data);
        renderDownloads(data.available_exports || []);
    })
    .catch(function(err) { console.error('Failed to load stats:', err); });

    // Load taxonomy tree and concept-data-points mapping in parallel
    Promise.all([
        fetch('data/taxonomy.json').then(function(r) { return r.ok ? r.json() : null; }),
        fetch('data/concept-data-points.json').then(function(r) { return r.ok ? r.json() : null; })
    ])
    .then(function(results) {
        var taxonomyData = results[0];
        var dataPointsMap = results[1];
        window._conceptDataPoints = dataPointsMap || {};
        if (taxonomyData && taxonomyData.taxonomy) {
            renderTaxonomyTree(taxonomyData.taxonomy);
        }
    })
    .catch(function() {});
}

function renderResultsStats(data) {
    var grid = document.getElementById('resultsStatsGrid');
    grid.innerHTML = '';

    var items = [];
    if (data.extraction) {
        items.push(['Concepts', data.extraction.concepts]);
        items.push(['Entities', data.extraction.entities]);
        items.push(['Relationships', data.extraction.relationships]);
    }
    if (data.deduplicated) {
        items.push(['Unique Concepts', data.deduplicated.concepts]);
        items.push(['Unique Entities', data.deduplicated.entities]);
        items.push(['Merges', data.deduplicated.merges]);
    }
    if (data.graph) {
        items.push(['Graph Nodes', data.graph.nodes]);
        items.push(['Graph Edges', data.graph.edges]);
    }
    if (data.crosslinks) {
        items.push(['New Links', data.crosslinks.new_relationships]);
    }

    items.forEach(function(item) {
        grid.innerHTML += '<div class="stat-card"><div class="stat-value">' + (item[1] != null ? item[1].toLocaleString() : '-') + '</div><div class="stat-label">' + item[0] + '</div></div>';
    });
}

function renderTopConcepts(concepts) {
    var table = document.getElementById('topConceptsTable');
    if (concepts.length === 0) {
        table.innerHTML = '<tr><td class="text-dim">No concepts extracted yet.</td></tr>';
        return;
    }
    var html = '<thead><tr><th>Concept</th><th>Category</th><th>Frequency</th><th>Description</th></tr></thead><tbody>';
    concepts.forEach(function(c) {
        html += '<tr><td style="font-weight:600;color:var(--text-bright)">' + escapeHtml(c.name || '') + '</td>'
            + '<td>' + escapeHtml(c.category || '') + '</td>'
            + '<td style="color:var(--primary);font-weight:700">' + (c.frequency || 0) + '</td>'
            + '<td class="text-dim">' + escapeHtml((c.description || '').substring(0, 80)) + '</td></tr>';
    });
    html += '</tbody>';
    table.innerHTML = html;
}

function renderEntitiesTable(data) {
    var table = document.getElementById('entitiesTable');
    if (!data.entities_by_type) {
        table.innerHTML = '<tr><td class="text-dim">No entities extracted yet.</td></tr>';
        return;
    }
    var html = '<thead><tr><th>Entity Type</th><th>Count</th></tr></thead><tbody>';
    var types = data.entities_by_type;
    var sorted = Object.keys(types).sort(function(a, b) { return types[b] - types[a]; });
    sorted.forEach(function(t) {
        html += '<tr><td style="font-weight:600;color:var(--text-bright)">' + escapeHtml(t) + '</td>'
            + '<td style="color:var(--primary);font-weight:700">' + types[t] + '</td></tr>';
    });
    html += '</tbody>';
    table.innerHTML = html;
}

function renderDownloads(exports) {
    var grid = document.getElementById('downloadsGrid');
    grid.innerHTML = '';

    var formatInfo = {
        'knowledge_graph.json': { icon: '{ }', desc: 'Node-link format' },
        'knowledge_graph.graphml': { icon: '&#x1F4C8;', desc: 'For Gephi / Cytoscape' },
        'relationships.csv': { icon: '&#x1F4CB;', desc: 'Tabular relationships' },
        'taxonomy_tree.md': { icon: '&#x1F4DD;', desc: 'Taxonomy as Markdown' },
        'knowledge_graph.html': { icon: '&#x1F310;', desc: 'Interactive vis.js graph' },
        'summary_report.html': { icon: '&#x1F4CA;', desc: 'Full HTML summary report' },
        'knowledge_graph.md': { icon: '&#x1F4D6;', desc: 'Graph summary (Markdown)' },
        'taxonomy.json': { icon: '&#x1F333;', desc: 'Taxonomy hierarchy' },
        'extractions.json': { icon: '&#x1F9EC;', desc: 'Raw extractions (backup)' },
        'crosslinks.json': { icon: '&#x1F517;', desc: 'Cross-domain relationships' },
        'obsidian_vault.zip': { icon: '&#x1F5C2;', desc: 'Obsidian-compatible vault' },
    };

    exports.forEach(function(exp) {
        var info = formatInfo[exp.name] || { icon: '&#x1F4C4;', desc: '' };
        var href = 'downloads/' + exp.name;

        grid.innerHTML += '<a class="download-item" href="' + href + '" download>'
            + '<span class="download-icon">' + info.icon + '</span>'
            + '<div><div class="download-label">' + escapeHtml(exp.label) + '</div>'
            + '<div class="download-desc">' + info.desc + '</div></div></a>';
    });
}

// ======================================================================
// Taxonomy Tree Rendering
// ======================================================================
function renderTaxonomyTree(taxonomy) {
    var container = document.getElementById('taxonomyTree');
    container.innerHTML = '';

    if (!taxonomy || taxonomy.length === 0) {
        container.innerHTML = '<div class="text-dim">No taxonomy data available.</div>';
        return;
    }

    taxonomy.forEach(function(domain) {
        var node = createTreeNode(domain, true);
        container.appendChild(node);
    });
}

function createTreeNode(item, isDomain) {
    var li = document.createElement('div');
    li.className = 'tree-node' + (isDomain ? ' tree-domain' : '');

    var hasChildren = item.children && item.children.length > 0;
    var isLeaf = !hasChildren && item.frequency;

    var content = document.createElement('div');
    content.className = 'tree-node-content';

    var toggle = document.createElement('span');
    toggle.className = 'tree-toggle' + (hasChildren ? '' : ' leaf');
    toggle.innerHTML = '&#x25B6;';
    content.appendChild(toggle);

    var name = document.createElement('span');
    name.className = 'tree-name';
    name.textContent = item.name || 'Unknown';
    content.appendChild(name);

    if (item.frequency) {
        var freq = document.createElement('span');
        freq.className = 'tree-freq';
        freq.textContent = item.frequency;
        content.appendChild(freq);
    }

    if (item.description && !isDomain) {
        var desc = document.createElement('span');
        desc.className = 'tree-desc';
        desc.textContent = '-- ' + item.description.substring(0, 60);
        content.appendChild(desc);
    }

    // Data points indicator for leaf nodes
    if (isLeaf && window._conceptDataPoints) {
        var conceptKey = (item.name || '').trim().toLowerCase();
        var dataPoints = window._conceptDataPoints[conceptKey];
        if (dataPoints && dataPoints.length > 0) {
            var dpIndicator = document.createElement('span');
            dpIndicator.className = 'tree-data-points';
            dpIndicator.textContent = '📋 ' + dataPoints.length + ' data point' + (dataPoints.length !== 1 ? 's' : '');

            // Build the expandable panel (hidden by default)
            var panel = document.createElement('div');
            panel.className = 'tree-data-panel';

            var MAX_SHOW = 20;
            var toShow = dataPoints.slice(0, MAX_SHOW);
            toShow.forEach(function(rid) {
                var badge = document.createElement('span');
                badge.className = 'result-id-badge';
                badge.textContent = '#' + rid;
                panel.appendChild(badge);
            });
            if (dataPoints.length > MAX_SHOW) {
                var more = document.createElement('span');
                more.className = 'tree-data-more';
                more.textContent = '... and ' + (dataPoints.length - MAX_SHOW) + ' more';
                panel.appendChild(more);
            }

            dpIndicator.addEventListener('click', function(e) {
                e.stopPropagation();
                panel.classList.toggle('open');
                dpIndicator.classList.toggle('active');
            });

            content.appendChild(dpIndicator);
            li.appendChild(content);
            li.appendChild(panel);
            return li;
        }
    }

    li.appendChild(content);

    if (hasChildren) {
        var childContainer = document.createElement('div');
        childContainer.className = 'tree-children collapsed';

        item.children.forEach(function(child) {
            childContainer.appendChild(createTreeNode(child, false));
        });

        li.appendChild(childContainer);

        content.addEventListener('click', function() {
            toggle.classList.toggle('expanded');
            childContainer.classList.toggle('collapsed');
        });
    }

    return li;
}

// ======================================================================
// Visualizations (D3.js)
// ======================================================================
var vizLoaded = false;
var CGIAR_COLORS = [
    '#00843D', '#0077B6', '#F77F00', '#00B4D8', '#7B2D8E',
    '#FCBF49', '#E63946', '#2D6A4F', '#264653', '#E9C46A', '#023E8A'
];

function loadVisualizations() {
    if (vizLoaded) return;

    fetch('data/stats.json')
    .then(function(r) { return r.json(); })
    .then(function(data) {
        if (!data.has_results) {
            document.getElementById('vizNoData').classList.remove('hidden');
            document.getElementById('vizContent').classList.add('hidden');
            return;
        }
        document.getElementById('vizNoData').classList.add('hidden');
        document.getElementById('vizContent').classList.remove('hidden');
        vizLoaded = true;

        buildSunburst();
        buildNetworkGraph();
        buildTopConceptsBar(data);
        buildEntityDonut(data);
        buildRelTypesBar();
        buildDomainBar();
    })
    .catch(function() {
        document.getElementById('vizNoData').classList.remove('hidden');
        document.getElementById('vizContent').classList.add('hidden');
    });
}

// ---- Tooltip helper ----
function createTooltip() {
    var tip = d3.select('body').append('div')
        .attr('class', 'viz-tooltip')
        .style('display', 'none');
    return tip;
}

// ---- 1. Zoomable Sunburst (improved) ----
function buildSunburst() {
    fetch('data/taxonomy.json')
    .then(function(r) { if (!r.ok) throw new Error('No taxonomy'); return r.json(); })
    .then(function(data) {
        var taxonomy = data.taxonomy;
        if (!taxonomy || taxonomy.length === 0) {
            document.getElementById('sunburstContainer').innerHTML = '<div class="viz-empty">No taxonomy data available.</div>';
            return;
        }

        var container = document.getElementById('sunburstContainer');
        container.innerHTML = '';
        var width = container.clientWidth;
        var height = Math.min(width, 650);
        var radius = Math.min(width, height) / 2;

        // Build hierarchy
        var root = { name: 'CGIAR Taxonomy', children: taxonomy };

        var hierarchy = d3.hierarchy(root)
            .sum(function(d) { return (!d.children || d.children.length === 0) ? (d.frequency || 1) : 0; })
            .sort(function(a, b) { return b.value - a.value; });

        var partitionLayout = d3.partition().size([2 * Math.PI, radius]);
        partitionLayout(hierarchy);

        // Color scale: map top-level domains to CGIAR colors
        var domainNames = taxonomy.map(function(d) { return d.name; });
        var colorScale = d3.scaleOrdinal().domain(domainNames).range(CGIAR_COLORS);

        // getColor: vary hue within each domain so siblings are distinguishable
        function getColor(d) {
            var node = d;
            while (node.depth > 1) node = node.parent;
            if (node.depth === 0) return '#333';
            var base = d3.hsl(colorScale(node.data.name));
            if (!base || isNaN(base.h)) return '#555';

            var siblingIndex = 0;
            if (d.parent && d.parent.children) {
                siblingIndex = d.parent.children.indexOf(d);
            }

            var nameHash = 0;
            var name = d.data.name || '';
            for (var ci = 0; ci < name.length; ci++) {
                nameHash = ((nameHash << 5) - nameHash + name.charCodeAt(ci)) | 0;
            }
            nameHash = Math.abs(nameHash) % 360;

            var hueShift = (siblingIndex * 23 + nameHash * 0.08) % 360;
            base.h = (base.h + hueShift * 0.18) % 360;

            var depthFactor = Math.max(0, d.depth - 1);
            base.s = Math.max(0.35, Math.min(0.95, base.s - depthFactor * 0.06));
            base.l = Math.min(0.80, Math.max(0.30, base.l + depthFactor * 0.05));

            return base.toString();
        }

        var svg = d3.select(container).append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', [-width/2, -height/2, width, height].join(' '));

        var arc = d3.arc()
            .startAngle(function(d) { return d.x0; })
            .endAngle(function(d) { return d.x1; })
            .padAngle(function(d) { return Math.min((d.x1 - d.x0) / 2, 0.005); })
            .padRadius(radius / 2)
            .innerRadius(function(d) { return d.y0; })
            .outerRadius(function(d) { return d.y1 - 1; });

        var tip = createTooltip();

        // Current root for zooming
        var currentRoot = hierarchy;

        // Helper: check if d is a descendant of root (or is root itself)
        function isDescendantOf(d, root) {
            var n = d;
            while (n) {
                if (n === root) return true;
                n = n.parent;
            }
            return false;
        }

        // Helper: compute the correct fill-opacity for a segment given the current zoom root
        function segmentOpacity(d) {
            var relDepth = d.depth - currentRoot.depth;
            if (!isDescendantOf(d, currentRoot)) return 0;
            if (relDepth <= 0) return 0;
            return relDepth <= 3 ? (1 - relDepth * 0.10) : 0.35;
        }

        // Helper: compute stroke visibility
        function segmentStroke(d) {
            if (!isDescendantOf(d, currentRoot)) return 'none';
            var angularSize = d.x1 - d.x0;
            if (angularSize < 0.015) return 'none';
            return 'rgba(26,26,46,0.6)';
        }
        function segmentStrokeWidth(d) {
            if (!isDescendantOf(d, currentRoot)) return 0;
            var angularSize = d.x1 - d.x0;
            if (angularSize < 0.015) return 0;
            return 0.5;
        }

        var paths = svg.selectAll('path')
            .data(hierarchy.descendants().filter(function(d) { return d.depth > 0; }))
            .join('path')
            .attr('d', arc)
            .attr('fill', getColor)
            .attr('fill-opacity', function(d) { return segmentOpacity(d); })
            .attr('stroke', function(d) { return segmentStroke(d); })
            .attr('stroke-width', function(d) { return segmentStrokeWidth(d); })
            .style('cursor', function(d) { return d.children ? 'pointer' : 'default'; })
            .on('mouseover', function(event, d) {
                if (segmentOpacity(d) === 0) return;
                d3.select(this)
                    .attr('fill-opacity', 1)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 1.5);
                var freq = d.data.frequency ? ' (freq: ' + d.data.frequency + ')' : '';
                var val = ' | size: ' + d.value;
                tip.style('display', 'block')
                    .html('<strong>' + escapeHtml(d.data.name) + '</strong>' + freq + val
                        + (d.data.description ? '<br><span style="color:#8892a4">' + escapeHtml(d.data.description) + '</span>' : ''));
            })
            .on('mousemove', function(event) {
                tip.style('left', (event.pageX + 14) + 'px').style('top', (event.pageY - 14) + 'px');
            })
            .on('mouseout', function(event, d) {
                d3.select(this)
                    .attr('fill-opacity', segmentOpacity(d))
                    .attr('stroke', segmentStroke(d))
                    .attr('stroke-width', segmentStrokeWidth(d));
                tip.style('display', 'none');
            })
            .on('click', function(event, d) {
                if (d.children) zoomTo(d);
            });

        // Center label
        var centerText = svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.3em')
            .attr('fill', '#e0e0e0')
            .attr('font-size', '13px')
            .attr('font-weight', '600')
            .style('cursor', 'pointer')
            .text('CGIAR Taxonomy')
            .on('click', function() {
                zoomTo(hierarchy);
            });

        // Reset button wiring
        var resetBtn = document.getElementById('sunburstResetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', function() {
                zoomTo(hierarchy);
            });
        }

        // Count descendants for the segment counter display
        function countDescendants(node) {
            var count = 0;
            if (node.children) {
                node.children.forEach(function(c) {
                    count += 1;
                    count += countDescendants(c);
                });
            }
            return count;
        }

        // Update the navigation bar: breadcrumb, reset button, segment counter
        function updateNavBar(target) {
            var breadcrumb = [];
            var node = target;
            while (node) {
                breadcrumb.unshift(node);
                node = node.parent;
            }

            var bcEl = document.getElementById('sunburstBreadcrumbNew');
            if (bcEl) {
                bcEl.innerHTML = '';
                breadcrumb.forEach(function(b, i) {
                    if (i > 0) {
                        var sep = document.createElement('span');
                        sep.className = 'sunburst-breadcrumb-sep';
                        sep.textContent = ' / ';
                        bcEl.appendChild(sep);
                    }
                    var label = b.data.name || 'Root';
                    if (i < breadcrumb.length - 1) {
                        var link = document.createElement('span');
                        link.textContent = label;
                        link.setAttribute('data-depth', b.depth);
                        link.addEventListener('click', function() {
                            vizSunburstZoom(parseInt(this.getAttribute('data-depth'), 10));
                        });
                        bcEl.appendChild(link);
                    } else {
                        var current = document.createElement('span');
                        current.className = 'sunburst-breadcrumb-current';
                        current.textContent = label;
                        bcEl.appendChild(current);
                    }
                });
            }

            // Show/hide reset button
            if (resetBtn) {
                if (target === hierarchy) {
                    resetBtn.classList.remove('sunburst-visible');
                } else {
                    resetBtn.classList.add('sunburst-visible');
                }
            }

            // Segment counter
            var segCount = target.children ? countDescendants(target) : 0;
            var countEl = document.getElementById('sunburstSegmentCount');
            if (countEl) {
                countEl.textContent = segCount > 0 ? segCount + ' segment' + (segCount !== 1 ? 's' : '') : '';
            }
        }

        function zoomTo(target) {
            currentRoot = target;

            // Update the navigation bar (breadcrumb + reset + counter)
            updateNavBar(target);

            // Also clear the legacy breadcrumb element
            var bc = document.getElementById('sunburstBreadcrumb');
            if (bc) bc.innerHTML = '';

            // Transition
            var targetX0 = target.x0;
            var targetX1 = target.x1;
            var targetY0 = target.y0;

            var xScale = d3.scaleLinear().domain([targetX0, targetX1]).range([0, 2 * Math.PI]);
            var yScale = d3.scaleLinear().domain([targetY0, radius]).range([0, radius]);

            paths.transition().duration(750)
                .attrTween('d', function(d) {
                    var interpX0 = d3.interpolate(d._x0 || d.x0, xScale(d.x0));
                    var interpX1 = d3.interpolate(d._x1 || d.x1, xScale(d.x1));
                    var interpY0 = d3.interpolate(d._y0 || d.y0, yScale(d.y0));
                    var interpY1 = d3.interpolate(d._y1 || d.y1, yScale(d.y1));
                    return function(t) {
                        d._x0 = interpX0(t);
                        d._x1 = interpX1(t);
                        d._y0 = interpY0(t);
                        d._y1 = interpY1(t);
                        return d3.arc()
                            .startAngle(d._x0).endAngle(d._x1)
                            .innerRadius(d._y0).outerRadius(d._y1 - 1)
                            .padAngle(Math.min((d._x1 - d._x0) / 2, 0.005))
                            .padRadius(radius / 2)();
                    };
                })
                .attr('fill-opacity', function(d) {
                    if (!isDescendantOf(d, target)) return 0;
                    var relDepth = d.depth - target.depth;
                    if (relDepth <= 0) return 0;
                    return relDepth <= 3 ? (1 - relDepth * 0.10) : 0.35;
                })
                .attr('stroke', function(d) {
                    if (!isDescendantOf(d, target)) return 'none';
                    var relDepth = d.depth - target.depth;
                    if (relDepth <= 0) return 'none';
                    var angularSize = d.x1 - d.x0;
                    var scaledSize = angularSize / (target.x1 - target.x0) * 2 * Math.PI;
                    if (scaledSize < 0.015) return 'none';
                    return 'rgba(26,26,46,0.6)';
                })
                .attr('stroke-width', function(d) {
                    if (!isDescendantOf(d, target)) return 0;
                    var relDepth = d.depth - target.depth;
                    if (relDepth <= 0) return 0;
                    var angularSize = d.x1 - d.x0;
                    var scaledSize = angularSize / (target.x1 - target.x0) * 2 * Math.PI;
                    if (scaledSize < 0.015) return 0;
                    return 0.5;
                })
                .style('pointer-events', function(d) {
                    return isDescendantOf(d, target) && d.depth > target.depth ? 'all' : 'none';
                });

            centerText.text(target.data.name || 'CGIAR Taxonomy')
                .style('cursor', target === hierarchy ? 'default' : 'pointer');
        }

        // Initialize nav bar for root view
        updateNavBar(hierarchy);

        // Store zoom function globally for breadcrumb clicks
        window._sunburstHierarchy = hierarchy;
        var vizSunburstZoom = function(depth) {
            var node = currentRoot;
            var path = [];
            var n = node;
            while (n) { path.unshift(n); n = n.parent; }
            if (depth < path.length) zoomTo(path[depth]);
        };
        window.vizSunburstZoom = vizSunburstZoom;
    })
    .catch(function() {
        document.getElementById('sunburstContainer').innerHTML = '<div class="viz-empty">Could not load taxonomy data.</div>';
    });
}

// ---- 2. Force-Directed Network Graph (improved) ----
var _graphNodeLimit = 200;
var _graphFullData = null;
function buildNetworkGraph() {
    var dataPromise = _graphFullData
        ? Promise.resolve(_graphFullData)
        : fetch('data/graph-data.json').then(function(r) { if (!r.ok) throw new Error('No graph data'); return r.json(); });

    dataPromise
    .then(function(data) {
        // Cache full data for slider rebuilds
        if (!_graphFullData) _graphFullData = data;

        // Client-side node limiting: take top N nodes by frequency/degree
        var allNodes = data.nodes;
        var allLinks = data.links;
        var nodes, links;
        if (_graphNodeLimit >= allNodes.length) {
            nodes = allNodes.slice();
            links = allLinks.slice();
        } else {
            // Sort by frequency descending, take top N
            var sorted = allNodes.slice().sort(function(a, b) { return (b.frequency || 0) - (a.frequency || 0); });
            nodes = sorted.slice(0, _graphNodeLimit);
            var nodeIds = new Set(nodes.map(function(n) { return n.id; }));
            links = allLinks.filter(function(l) {
                var s = typeof l.source === 'object' ? l.source.id : l.source;
                var t = typeof l.target === 'object' ? l.target.id : l.target;
                return nodeIds.has(s) && nodeIds.has(t);
            });
        }

        if (!nodes || nodes.length === 0) {
            document.getElementById('networkContainer').innerHTML = '<div class="viz-empty">No graph data available.</div>';
            return;
        }

        var container = document.getElementById('networkContainer');
        container.innerHTML = '';
        var width = container.clientWidth;
        var height = 550;

        // ---- State ----
        var pinnedNodeId = null;
        var isPaused = false;
        var currentHops = 1;

        // ---- Color scale by category ----
        var categorySet = new Set();
        nodes.forEach(function(n) { categorySet.add(n.category || n.type); });
        var categories = Array.from(categorySet);
        var catColor = d3.scaleOrdinal().domain(categories).range(CGIAR_COLORS);

        // ---- Node size by degree ----
        var degreeMap = {};
        links.forEach(function(l) {
            var s = typeof l.source === 'object' ? l.source.id : l.source;
            var t = typeof l.target === 'object' ? l.target.id : l.target;
            degreeMap[s] = (degreeMap[s] || 0) + 1;
            degreeMap[t] = (degreeMap[t] || 0) + 1;
        });
        var maxDegree = d3.max(Object.values(degreeMap)) || 1;
        var rScale = d3.scaleSqrt().domain([0, maxDegree]).range([3, 18]);

        // ---- Node count badge ----
        var countEl = document.getElementById('graphNodeCount');
        if (countEl) {
            countEl.textContent = nodes.length + ' nodes, ' + links.length + ' connections';
        }

        // ---- Build a node lookup map ----
        var nodeMap = {};
        nodes.forEach(function(n) { nodeMap[n.id] = n; });

        // ---- SVG ----
        var svg = d3.select(container).append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', [0, 0, width, height].join(' '));

        var g = svg.append('g');
        var zoom = d3.zoom().scaleExtent([0.2, 5]).on('zoom', function(event) {
            g.attr('transform', event.transform);
        });
        svg.call(zoom);

        var tip = createTooltip();

        // ---- Simulation ----
        var simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(function(d) { return d.id; }).distance(80).strength(0.3))
            .force('charge', d3.forceManyBody().strength(-120))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(function(d) { return rScale(degreeMap[d.id] || 0) + 2; }));

        // ---- Links ----
        var link = g.append('g')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('stroke', 'rgba(255,255,255,0.08)')
            .attr('stroke-width', 1);

        // ---- Nodes ----
        var node = g.append('g')
            .selectAll('circle')
            .data(nodes)
            .join('circle')
            .attr('r', function(d) { return rScale(degreeMap[d.id] || 0); })
            .attr('fill', function(d) { return catColor(d.category || d.type); })
            .attr('fill-opacity', 0.85)
            .attr('stroke', 'rgba(255,255,255,0.2)')
            .attr('stroke-width', 0.5)
            .style('cursor', 'pointer')
            .call(d3.drag()
                .on('start', function(event, d) {
                    if (isPaused) return;
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                })
                .on('drag', function(event, d) {
                    if (isPaused) return;
                    d.fx = event.x; d.fy = event.y;
                })
                .on('end', function(event, d) {
                    if (isPaused) return;
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null; d.fy = null;
                }));

        // ---- Labels (show for degree >= 2) ----
        var labels = g.append('g')
            .selectAll('text')
            .data(nodes.filter(function(d) { return (degreeMap[d.id] || 0) >= 2; }))
            .join('text')
            .attr('font-size', '9px')
            .attr('fill', 'rgba(255,255,255,0.7)')
            .attr('text-anchor', 'middle')
            .attr('dy', function(d) { return rScale(degreeMap[d.id] || 0) + 11; })
            .text(function(d) { return d.label.length > 20 ? d.label.substring(0, 18) + '..' : d.label; })
            .style('pointer-events', 'none');

        // ================================================================
        // Multi-hop exploration helper
        // ================================================================
        function getNodesAtHops(nodeId, maxHops) {
            var hopMap = {};
            hopMap[nodeId] = 0;
            var frontier = [nodeId];

            for (var hop = 1; hop <= maxHops; hop++) {
                var nextFrontier = [];
                frontier.forEach(function(nid) {
                    links.forEach(function(l) {
                        var s = typeof l.source === 'object' ? l.source.id : l.source;
                        var t = typeof l.target === 'object' ? l.target.id : l.target;
                        if (s === nid && !(t in hopMap)) {
                            hopMap[t] = hop;
                            nextFrontier.push(t);
                        }
                        if (t === nid && !(s in hopMap)) {
                            hopMap[s] = hop;
                            nextFrontier.push(s);
                        }
                    });
                });
                frontier = nextFrontier;
            }
            return hopMap;
        }

        // Get the hop distance for a link given a hopMap
        function getLinkHop(l, hopMap) {
            var s = typeof l.source === 'object' ? l.source.id : l.source;
            var t = typeof l.target === 'object' ? l.target.id : l.target;
            var sHop = hopMap[s];
            var tHop = hopMap[t];
            if (sHop === undefined || tHop === undefined) return -1;
            return Math.max(sHop, tHop);
        }

        // ================================================================
        // Highlight / clear functions
        // ================================================================
        function applyHighlight(d) {
            pinnedNodeId = d.id;
            var hopMap = getNodesAtHops(d.id, currentHops);
            var nodeColor = catColor(d.category || d.type);

            // Nodes
            node.attr('fill-opacity', function(n) {
                    var hop = hopMap[n.id];
                    if (hop === undefined) return 0.06;
                    if (hop === 0) return 1;
                    if (hop === 1) return 1;
                    if (hop === 2) return 0.6;
                    return 0.35;
                })
                .attr('stroke', function(n) {
                    if (n.id === d.id) return '#ffffff';
                    return 'rgba(255,255,255,0.2)';
                })
                .attr('stroke-width', function(n) {
                    if (n.id === d.id) return 4;
                    return 0.5;
                })
                .attr('r', function(n) {
                    var base = rScale(degreeMap[n.id] || 0);
                    var hop = hopMap[n.id];
                    if (hop === undefined) return base;
                    if (hop === 0) return base * 1.2;
                    if (hop >= 3) return base * 0.85;
                    return base;
                });

            // Labels
            labels.attr('opacity', function(n) {
                var hop = hopMap[n.id];
                if (hop === undefined) return 0.08;
                if (hop <= 1) return 1;
                if (hop === 2) return 0.6;
                return 0.35;
            });

            // Links
            link.attr('stroke', function(l) {
                    var lh = getLinkHop(l, hopMap);
                    if (lh < 0) return 'rgba(255,255,255,0.02)';
                    if (lh <= 1) return nodeColor;
                    var s = typeof l.source === 'object' ? l.source.id : l.source;
                    var t = typeof l.target === 'object' ? l.target.id : l.target;
                    if (hopMap[s] === hopMap[t] && hopMap[s] >= 1) return '#F77F00';
                    if (lh === 2) return 'rgba(0,166,81,0.35)';
                    return 'rgba(0,166,81,0.2)';
                })
                .attr('stroke-width', function(l) {
                    var lh = getLinkHop(l, hopMap);
                    if (lh < 0) return 0.3;
                    if (lh <= 1) return 2.5;
                    if (lh === 2) return 1.5;
                    return 1;
                })
                .attr('stroke-opacity', function(l) {
                    var lh = getLinkHop(l, hopMap);
                    if (lh < 0) return 0.08;
                    if (lh <= 1) return 1;
                    if (lh === 2) return 0.7;
                    return 0.4;
                })
                .attr('stroke-dasharray', function(l) {
                    var lh = getLinkHop(l, hopMap);
                    if (lh <= 1) return null;
                    if (lh === 2) return '6,3';
                    return '3,3';
                });

            // Show detail panel
            showDetailPanel(d, hopMap);
        }

        function clearHighlight() {
            pinnedNodeId = null;

            node.attr('fill-opacity', 0.85)
                .attr('stroke', 'rgba(255,255,255,0.2)')
                .attr('stroke-width', 0.5)
                .attr('r', function(d) { return rScale(degreeMap[d.id] || 0); });

            labels.attr('opacity', 1);

            link.attr('stroke', 'rgba(255,255,255,0.08)')
                .attr('stroke-width', 1)
                .attr('stroke-opacity', 0.6)
                .attr('stroke-dasharray', null);

            hideDetailPanel();
        }

        // ================================================================
        // Detail panel
        // ================================================================
        function showDetailPanel(d, hopMap) {
            var panel = document.getElementById('graphDetailPanel');
            var content = document.getElementById('graphDetailContent');
            if (!panel || !content) return;
            var deg = degreeMap[d.id] || 0;
            var color = catColor(d.category || d.type);

            // Build connected entities list grouped by hop
            var connectionsByHop = {};
            Object.keys(hopMap).forEach(function(nid) {
                if (nid === d.id) return;
                var hop = hopMap[nid];
                if (!connectionsByHop[hop]) connectionsByHop[hop] = [];
                var n = nodeMap[nid];
                if (n) connectionsByHop[hop].push(n);
            });

            var html = '';
            html += '<div class="graph-detail-name">' + escapeHtml(d.label) + '</div>';
            html += '<div class="graph-detail-type">';
            html += escapeHtml(d.type || 'Unknown');
            if (d.category && d.category !== d.type) {
                html += ' <span class="graph-detail-type-badge" style="background:' + color + '22; color:' + color + '">' + escapeHtml(d.category) + '</span>';
            }
            html += '</div>';

            if (d.description) {
                html += '<div class="graph-detail-desc">' + escapeHtml(d.description) + '</div>';
            }

            html += '<div class="graph-detail-section">';
            html += '<div class="graph-detail-section-title">Properties</div>';
            html += '<div class="graph-detail-stat"><span class="graph-detail-stat-label">Connections</span><span class="graph-detail-stat-value">' + deg + '</span></div>';
            if (d.frequency && d.frequency > 1) {
                html += '<div class="graph-detail-stat"><span class="graph-detail-stat-label">Frequency</span><span class="graph-detail-stat-value">' + d.frequency + '</span></div>';
            }
            html += '</div>';

            // Connected entities sections by hop
            var hopLabels = {1: 'Direct connections', 2: '2nd-hop connections', 3: '3rd-hop connections'};
            for (var hop = 1; hop <= currentHops; hop++) {
                var conns = connectionsByHop[hop];
                if (!conns || conns.length === 0) continue;
                conns.sort(function(a, b) { return (degreeMap[b.id] || 0) - (degreeMap[a.id] || 0); });

                html += '<div class="graph-detail-section">';
                html += '<div class="graph-detail-section-title">' + hopLabels[hop] + ' (' + conns.length + ')</div>';
                html += '<ul class="graph-detail-connections">';
                conns.forEach(function(n) {
                    var nColor = catColor(n.category || n.type);
                    html += '<li class="graph-detail-conn-item" data-node-id="' + escapeHtml(n.id) + '">';
                    html += '<span class="graph-detail-conn-dot" style="background:' + nColor + '"></span>';
                    html += '<span>' + escapeHtml(n.label) + '</span>';
                    html += '<span class="graph-detail-conn-hop">' + (degreeMap[n.id] || 0) + ' conn.</span>';
                    html += '</li>';
                });
                html += '</ul>';
                html += '</div>';
            }

            content.innerHTML = html;
            panel.classList.add('graph-panel-open');

            // Make connection items clickable
            content.querySelectorAll('.graph-detail-conn-item').forEach(function(item) {
                item.addEventListener('click', function() {
                    var targetId = this.getAttribute('data-node-id');
                    var targetNode = nodeMap[targetId];
                    if (targetNode) {
                        applyHighlight(targetNode);
                        if (targetNode.x && targetNode.y) {
                            svg.transition().duration(400).call(
                                zoom.transform,
                                d3.zoomIdentity.translate(width / 2, height / 2).scale(1.5).translate(-targetNode.x, -targetNode.y)
                            );
                        }
                    }
                });
            });
        }

        function hideDetailPanel() {
            var panel = document.getElementById('graphDetailPanel');
            if (panel) panel.classList.remove('graph-panel-open');
        }

        // ================================================================
        // Mouse events on nodes
        // ================================================================

        // Hover (only when no node is pinned)
        node.on('mouseover', function(event, d) {
            if (pinnedNodeId) return;
            var connected = new Set();
            links.forEach(function(l) {
                var s = typeof l.source === 'object' ? l.source.id : l.source;
                var t = typeof l.target === 'object' ? l.target.id : l.target;
                if (s === d.id) connected.add(t);
                if (t === d.id) connected.add(s);
            });
            connected.add(d.id);

            node.attr('fill-opacity', function(n) { return connected.has(n.id) ? 1 : 0.1; });
            link.attr('stroke', function(l) {
                var s = typeof l.source === 'object' ? l.source.id : l.source;
                var t = typeof l.target === 'object' ? l.target.id : l.target;
                return (s === d.id || t === d.id) ? 'rgba(0,166,81,0.6)' : 'rgba(255,255,255,0.03)';
            }).attr('stroke-width', function(l) {
                var s = typeof l.source === 'object' ? l.source.id : l.source;
                var t = typeof l.target === 'object' ? l.target.id : l.target;
                return (s === d.id || t === d.id) ? 2 : 0.5;
            });

            var deg = degreeMap[d.id] || 0;
            tip.style('display', 'block')
                .html('<strong>' + escapeHtml(d.label) + '</strong>'
                    + '<br>Type: ' + escapeHtml(d.type) + (d.category ? ' / ' + escapeHtml(d.category) : '')
                    + '<br>Connections: ' + deg
                    + (d.frequency > 1 ? '<br>Frequency: ' + d.frequency : '')
                    + (d.description ? '<br><span style="color:#8892a4">' + escapeHtml(d.description) + '</span>' : ''));
        });

        node.on('mousemove', function(event) {
            if (pinnedNodeId) return;
            tip.style('left', (event.pageX + 14) + 'px').style('top', (event.pageY - 14) + 'px');
        });

        node.on('mouseout', function() {
            if (pinnedNodeId) return;
            node.attr('fill-opacity', 0.85);
            link.attr('stroke', 'rgba(255,255,255,0.08)').attr('stroke-width', 1);
            tip.style('display', 'none');
        });

        // Click to stick
        node.on('click', function(event, d) {
            event.stopPropagation();
            tip.style('display', 'none');
            applyHighlight(d);
        });

        // Click SVG background to clear pin
        svg.on('click', function(event) {
            var target = event.target;
            if (target === svg.node() || target.tagName.toLowerCase() === 'svg') {
                clearHighlight();
            }
        });

        // ================================================================
        // Tick
        // ================================================================
        simulation.on('tick', function() {
            link.attr('x1', function(d) { return d.source.x; })
                .attr('y1', function(d) { return d.source.y; })
                .attr('x2', function(d) { return d.target.x; })
                .attr('y2', function(d) { return d.target.y; });
            node.attr('cx', function(d) { return d.x; })
                .attr('cy', function(d) { return d.y; });
            labels.attr('x', function(d) { return d.x; })
                .attr('y', function(d) { return d.y; });
        });

        // ================================================================
        // Search functionality
        // ================================================================
        document.getElementById('graphSearch').addEventListener('input', function(e) {
            var query = e.target.value.toLowerCase().trim();
            if (!query) {
                if (pinnedNodeId) {
                    var pNode = nodeMap[pinnedNodeId];
                    if (pNode) { applyHighlight(pNode); return; }
                }
                node.attr('fill-opacity', 0.85).attr('r', function(d) { return rScale(degreeMap[d.id] || 0); });
                link.attr('stroke', 'rgba(255,255,255,0.08)').attr('stroke-width', 1);
                return;
            }

            var matched = new Set();
            var connectedToMatch = new Set();
            nodes.forEach(function(n) {
                if (n.label.toLowerCase().indexOf(query) !== -1) matched.add(n.id);
            });
            links.forEach(function(l) {
                var s = typeof l.source === 'object' ? l.source.id : l.source;
                var t = typeof l.target === 'object' ? l.target.id : l.target;
                if (matched.has(s)) connectedToMatch.add(t);
                if (matched.has(t)) connectedToMatch.add(s);
            });

            node.attr('fill-opacity', function(d) {
                if (matched.has(d.id)) return 1;
                if (connectedToMatch.has(d.id)) return 0.5;
                return 0.05;
            }).attr('r', function(d) {
                var base = rScale(degreeMap[d.id] || 0);
                return matched.has(d.id) ? base * 1.5 : base;
            });

            link.attr('stroke', function(l) {
                var s = typeof l.source === 'object' ? l.source.id : l.source;
                var t = typeof l.target === 'object' ? l.target.id : l.target;
                if (matched.has(s) || matched.has(t)) return 'rgba(0,166,81,0.5)';
                return 'rgba(255,255,255,0.02)';
            }).attr('stroke-width', function(l) {
                var s = typeof l.source === 'object' ? l.source.id : l.source;
                var t = typeof l.target === 'object' ? l.target.id : l.target;
                return (matched.has(s) || matched.has(t)) ? 2 : 0.5;
            });

            if (matched.size > 0) {
                var firstMatch = nodes.find(function(n) { return matched.has(n.id); });
                if (firstMatch && firstMatch.x && firstMatch.y) {
                    svg.transition().duration(500).call(
                        zoom.transform,
                        d3.zoomIdentity.translate(width/2, height/2).scale(1.5).translate(-firstMatch.x, -firstMatch.y)
                    );
                }
            }
        });

        // ================================================================
        // Pause / Resume button
        // ================================================================
        var pauseBtn = document.getElementById('graphPauseBtn');
        if (pauseBtn) {
            pauseBtn.addEventListener('click', function() {
                if (isPaused) {
                    isPaused = false;
                    simulation.alpha(0.3).restart();
                    pauseBtn.innerHTML = '&#x23F8; Pause';
                    pauseBtn.classList.remove('graph-btn-active');
                } else {
                    isPaused = true;
                    simulation.stop();
                    pauseBtn.innerHTML = '&#x25B6; Play';
                    pauseBtn.classList.add('graph-btn-active');
                }
            });
        }

        // ================================================================
        // Hops slider
        // ================================================================
        var hopsSlider = document.getElementById('graphHopsSlider');
        var hopsValue = document.getElementById('graphHopsValue');
        if (hopsSlider) {
            hopsSlider.addEventListener('input', function() {
                currentHops = parseInt(this.value, 10);
                if (hopsValue) hopsValue.textContent = currentHops;
                if (pinnedNodeId) {
                    var pNode = nodeMap[pinnedNodeId];
                    if (pNode) applyHighlight(pNode);
                }
            });
        }

        // ================================================================
        // Reset button
        // ================================================================
        var graphResetBtn = document.getElementById('graphResetBtn');
        if (graphResetBtn) {
            graphResetBtn.addEventListener('click', function() {
                clearHighlight();
                svg.transition().duration(400).call(
                    zoom.transform,
                    d3.zoomIdentity
                );
                var searchInput = document.getElementById('graphSearch');
                if (searchInput) {
                    searchInput.value = '';
                    node.attr('fill-opacity', 0.85).attr('r', function(d) { return rScale(degreeMap[d.id] || 0); });
                    link.attr('stroke', 'rgba(255,255,255,0.08)').attr('stroke-width', 1).attr('stroke-dasharray', null);
                }
                currentHops = 1;
                if (hopsSlider) hopsSlider.value = '1';
                if (hopsValue) hopsValue.textContent = '1';
            });
        }

        // ================================================================
        // Detail panel close button
        // ================================================================
        var detailClose = document.getElementById('graphDetailClose');
        if (detailClose) {
            detailClose.addEventListener('click', function() {
                clearHighlight();
            });
        }

        // ================================================================
        // Keyboard: Escape unpins and closes panel
        // ================================================================
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && pinnedNodeId) {
                clearHighlight();
            }
        });

        // ================================================================
        // Node limit slider — rebuilds the graph with more/fewer nodes
        // ================================================================
        var nodeSlider = document.getElementById('graphNodeSlider');
        var nodeSliderValue = document.getElementById('graphNodeSliderValue');
        if (nodeSlider) {
            nodeSlider.addEventListener('change', function() {
                var newLimit = parseInt(this.value, 10);
                if (nodeSliderValue) nodeSliderValue.textContent = newLimit;
                _graphNodeLimit = newLimit;
                if (simulation) simulation.stop();
                buildNetworkGraph();
            });
            nodeSlider.addEventListener('input', function() {
                if (nodeSliderValue) nodeSliderValue.textContent = this.value;
            });
        }

        // ================================================================
        // Link tooltip on hover
        // ================================================================
        link.on('mouseover', function(event, d) {
            var s = typeof d.source === 'object' ? d.source : nodeMap[d.source];
            var t = typeof d.target === 'object' ? d.target : nodeMap[d.target];
            var relType = d.type || d.relationship || 'related_to';
            if (s && t) {
                tip.style('display', 'block')
                    .html('<strong>' + escapeHtml(s.label || s.id) + '</strong>'
                        + ' <span style="color:var(--primary)">&rarr; ' + escapeHtml(relType) + ' &rarr;</span> '
                        + '<strong>' + escapeHtml(t.label || t.id) + '</strong>');
            }
        }).on('mousemove', function(event) {
            tip.style('left', (event.pageX + 14) + 'px').style('top', (event.pageY - 14) + 'px');
        }).on('mouseout', function() {
            tip.style('display', 'none');
        });

        // ================================================================
        // Legend
        // ================================================================
        var legend = svg.append('g').attr('transform', 'translate(12, 12)');
        var displayCats = categories.slice(0, 8);
        displayCats.forEach(function(cat, i) {
            var row = legend.append('g').attr('transform', 'translate(0,' + (i * 18) + ')');
            row.append('circle').attr('r', 5).attr('cx', 5).attr('cy', 0).attr('fill', catColor(cat));
            row.append('text').attr('x', 14).attr('y', 4).attr('font-size', '10px').attr('fill', '#8892a4').text(cat);
        });
    })
    .catch(function() {
        document.getElementById('networkContainer').innerHTML = '<div class="viz-empty">Could not load graph data.</div>';
    });
}

// ---- 3. Top Concepts Bar Chart ----
function buildTopConceptsBar(statsData) {
    var concepts = (statsData.top_concepts || []).slice(0, 30);
    if (concepts.length === 0) {
        document.getElementById('topConceptsBarContainer').innerHTML = '<div class="viz-empty">No concept data.</div>';
        return;
    }

    var container = document.getElementById('topConceptsBarContainer');
    container.innerHTML = '';
    var width = container.clientWidth;
    var barHeight = 22;
    var margin = { top: 10, right: 60, bottom: 20, left: 200 };
    var chartHeight = concepts.length * barHeight;
    var height = chartHeight + margin.top + margin.bottom;

    concepts.forEach(function(c) {
        c._label = c.name.length > 28 ? c.name.substring(0, 26) + '..' : c.name;
    });

    var svg = d3.select(container).append('svg')
        .attr('width', width)
        .attr('height', height);

    var xScale = d3.scaleLinear()
        .domain([0, d3.max(concepts, function(d) { return d.frequency; })])
        .range([0, width - margin.left - margin.right]);

    var yScale = d3.scaleBand()
        .domain(concepts.map(function(d) { return d._label; }))
        .range([margin.top, margin.top + chartHeight])
        .padding(0.25);

    var g = svg.append('g');

    var defs = svg.append('defs');
    var grad = defs.append('linearGradient').attr('id', 'barGrad').attr('x1', '0%').attr('x2', '100%');
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#00843D');
    grad.append('stop').attr('offset', '100%').attr('stop-color', '#00B4D8');

    g.selectAll('.bar-label')
        .data(concepts)
        .join('text')
        .attr('x', margin.left - 8)
        .attr('y', function(d) { return yScale(d._label) + yScale.bandwidth() / 2 + 4; })
        .attr('text-anchor', 'end')
        .attr('font-size', '11px')
        .attr('fill', '#e0e0e0')
        .text(function(d) { return d._label; });

    g.selectAll('.bar')
        .data(concepts)
        .join('rect')
        .attr('x', margin.left)
        .attr('y', function(d) { return yScale(d._label); })
        .attr('height', yScale.bandwidth())
        .attr('rx', 3)
        .attr('fill', 'url(#barGrad)')
        .attr('fill-opacity', 0.85)
        .attr('width', 0)
        .transition().duration(800).delay(function(d, i) { return i * 25; })
        .attr('width', function(d) { return xScale(d.frequency); });

    g.selectAll('.bar-value')
        .data(concepts)
        .join('text')
        .attr('x', function(d) { return margin.left + xScale(d.frequency) + 6; })
        .attr('y', function(d) { return yScale(d._label) + yScale.bandwidth() / 2 + 4; })
        .attr('font-size', '11px')
        .attr('fill', '#00A651')
        .attr('font-weight', '600')
        .attr('opacity', 0)
        .text(function(d) { return d.frequency; })
        .transition().duration(400).delay(function(d, i) { return 400 + i * 25; })
        .attr('opacity', 1);
}

// ---- 4. Entity Type Distribution Donut ----
function buildEntityDonut(statsData) {
    var entityTypes = statsData.entities_by_type;
    if (!entityTypes || Object.keys(entityTypes).length === 0) {
        document.getElementById('entityDonutContainer').innerHTML = '<div class="viz-empty">No entity data.</div>';
        return;
    }

    var container = document.getElementById('entityDonutContainer');
    container.innerHTML = '';
    var width = container.clientWidth;
    var height = 380;
    var radius = Math.min(width, height) / 2 - 20;

    var pieData = Object.keys(entityTypes).map(function(k) {
        return { type: k, count: entityTypes[k] };
    }).sort(function(a, b) { return b.count - a.count; });

    var total = d3.sum(pieData, function(d) { return d.count; });
    var colorScale = d3.scaleOrdinal().domain(pieData.map(function(d) { return d.type; })).range(CGIAR_COLORS);

    var svg = d3.select(container).append('svg')
        .attr('width', width)
        .attr('height', height);

    var g = svg.append('g').attr('transform', 'translate(' + (width / 2) + ',' + (height / 2) + ')');

    var pie = d3.pie().value(function(d) { return d.count; }).sort(null).padAngle(0.02);
    var arc = d3.arc().innerRadius(radius * 0.55).outerRadius(radius);
    var arcHover = d3.arc().innerRadius(radius * 0.55).outerRadius(radius + 8);

    var tip = createTooltip();

    g.selectAll('.slice')
        .data(pie(pieData))
        .join('path')
        .attr('d', arc)
        .attr('fill', function(d) { return colorScale(d.data.type); })
        .attr('fill-opacity', 0.85)
        .attr('stroke', 'rgba(26,26,46,0.5)')
        .attr('stroke-width', 1.5)
        .style('cursor', 'pointer')
        .on('mouseover', function(event, d) {
            d3.select(this).transition().duration(150).attr('d', arcHover).attr('fill-opacity', 1);
            var pct = ((d.data.count / total) * 100).toFixed(1);
            tip.style('display', 'block')
                .html('<strong>' + escapeHtml(d.data.type) + '</strong><br>' + d.data.count + ' entities (' + pct + '%)');
        })
        .on('mousemove', function(event) {
            tip.style('left', (event.pageX + 14) + 'px').style('top', (event.pageY - 14) + 'px');
        })
        .on('mouseout', function() {
            d3.select(this).transition().duration(150).attr('d', arc).attr('fill-opacity', 0.85);
            tip.style('display', 'none');
        });

    g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '-0.2em')
        .attr('fill', '#fff')
        .attr('font-size', '22px')
        .attr('font-weight', '700')
        .text(total.toLocaleString());
    g.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '1.2em')
        .attr('fill', '#8892a4')
        .attr('font-size', '12px')
        .text('entities');

    var legend = svg.append('g').attr('transform', 'translate(12,' + (height - pieData.length * 16 - 4) + ')');
    pieData.forEach(function(d, i) {
        var row = legend.append('g').attr('transform', 'translate(0,' + (i * 16) + ')');
        row.append('rect').attr('width', 10).attr('height', 10).attr('rx', 2).attr('fill', colorScale(d.type));
        var pct = ((d.count / total) * 100).toFixed(1);
        row.append('text').attr('x', 14).attr('y', 9).attr('font-size', '10px').attr('fill', '#8892a4')
            .text(d.type + ' (' + d.count + ', ' + pct + '%)');
    });
}

// ---- 5. Relationship Types Bar Chart ----
function buildRelTypesBar() {
    fetch('data/relationship-types.json')
    .then(function(r) { if (!r.ok) throw new Error(); return r.json(); })
    .then(function(result) {
        var relTypes = result.types || {};
        if (Object.keys(relTypes).length === 0) {
            document.getElementById('relTypesBarContainer').innerHTML = '<div class="viz-empty">No relationship data.</div>';
            return;
        }

        var data = Object.keys(relTypes).map(function(k) {
            return { type: k, count: relTypes[k] };
        }).sort(function(a, b) { return b.count - a.count; }).slice(0, 20);

        var container = document.getElementById('relTypesBarContainer');
        container.innerHTML = '';
        var width = container.clientWidth;
        var barHeight = 24;
        var margin = { top: 10, right: 50, bottom: 20, left: 140 };
        var chartHeight = data.length * barHeight;
        var height = chartHeight + margin.top + margin.bottom;

        var svg = d3.select(container).append('svg')
            .attr('width', width)
            .attr('height', height);

        var xScale = d3.scaleLinear()
            .domain([0, d3.max(data, function(d) { return d.count; })])
            .range([0, width - margin.left - margin.right]);

        var yScale = d3.scaleBand()
            .domain(data.map(function(d) { return d.type; }))
            .range([margin.top, margin.top + chartHeight])
            .padding(0.25);

        var colorScale = d3.scaleOrdinal().domain(data.map(function(d) { return d.type; })).range(CGIAR_COLORS);

        svg.selectAll('.bar-label')
            .data(data)
            .join('text')
            .attr('x', margin.left - 8)
            .attr('y', function(d) { return yScale(d.type) + yScale.bandwidth() / 2 + 4; })
            .attr('text-anchor', 'end')
            .attr('font-size', '11px')
            .attr('fill', '#e0e0e0')
            .text(function(d) { return d.type; });

        svg.selectAll('.bar')
            .data(data)
            .join('rect')
            .attr('x', margin.left)
            .attr('y', function(d) { return yScale(d.type); })
            .attr('height', yScale.bandwidth())
            .attr('rx', 3)
            .attr('fill', function(d) { return colorScale(d.type); })
            .attr('fill-opacity', 0.8)
            .attr('width', 0)
            .transition().duration(700).delay(function(d, i) { return i * 30; })
            .attr('width', function(d) { return xScale(d.count); });

        svg.selectAll('.bar-value')
            .data(data)
            .join('text')
            .attr('x', function(d) { return margin.left + xScale(d.count) + 6; })
            .attr('y', function(d) { return yScale(d.type) + yScale.bandwidth() / 2 + 4; })
            .attr('font-size', '11px')
            .attr('fill', '#FCBF49')
            .attr('font-weight', '600')
            .text(function(d) { return d.count; })
            .attr('opacity', 0)
            .transition().duration(400).delay(function(d, i) { return 400 + i * 30; })
            .attr('opacity', 1);
    })
    .catch(function() {
        document.getElementById('relTypesBarContainer').innerHTML = '<div class="viz-empty">Could not load relationship data.</div>';
    });
}

// ---- 6. Domain Coverage Bar Chart ----
function buildDomainBar() {
    fetch('data/taxonomy.json')
    .then(function(r) { if (!r.ok) throw new Error(); return r.json(); })
    .then(function(data) {
        var taxonomy = data.taxonomy;
        if (!taxonomy || taxonomy.length === 0) {
            document.getElementById('domainBarContainer').innerHTML = '<div class="viz-empty">No taxonomy data.</div>';
            return;
        }

        function countConcepts(node) {
            if (!node.children || node.children.length === 0) return 1;
            var sum = 0;
            node.children.forEach(function(c) { sum += countConcepts(c); });
            return sum;
        }

        var domainData = taxonomy.map(function(d, i) {
            return { name: d.name, count: countConcepts(d), index: i };
        }).sort(function(a, b) { return b.count - a.count; });

        var container = document.getElementById('domainBarContainer');
        container.innerHTML = '';
        var width = container.clientWidth;
        var barHeight = 28;
        var margin = { top: 10, right: 50, bottom: 20, left: 220 };
        var chartHeight = domainData.length * barHeight;
        var height = chartHeight + margin.top + margin.bottom;

        var svg = d3.select(container).append('svg')
            .attr('width', width)
            .attr('height', height);

        var xScale = d3.scaleLinear()
            .domain([0, d3.max(domainData, function(d) { return d.count; })])
            .range([0, width - margin.left - margin.right]);

        var yScale = d3.scaleBand()
            .domain(domainData.map(function(d) { return d.name; }))
            .range([margin.top, margin.top + chartHeight])
            .padding(0.25);

        svg.selectAll('.bar-label')
            .data(domainData)
            .join('text')
            .attr('x', margin.left - 8)
            .attr('y', function(d) { return yScale(d.name) + yScale.bandwidth() / 2 + 4; })
            .attr('text-anchor', 'end')
            .attr('font-size', '11px')
            .attr('fill', '#e0e0e0')
            .text(function(d) {
                return d.name.length > 30 ? d.name.substring(0, 28) + '..' : d.name;
            });

        svg.selectAll('.bar')
            .data(domainData)
            .join('rect')
            .attr('x', margin.left)
            .attr('y', function(d) { return yScale(d.name); })
            .attr('height', yScale.bandwidth())
            .attr('rx', 4)
            .attr('fill', function(d) { return CGIAR_COLORS[d.index % CGIAR_COLORS.length]; })
            .attr('fill-opacity', 0.85)
            .attr('width', 0)
            .transition().duration(800).delay(function(d, i) { return i * 50; })
            .attr('width', function(d) { return xScale(d.count); });

        svg.selectAll('.bar-value')
            .data(domainData)
            .join('text')
            .attr('x', function(d) { return margin.left + xScale(d.count) + 6; })
            .attr('y', function(d) { return yScale(d.name) + yScale.bandwidth() / 2 + 4; })
            .attr('font-size', '11px')
            .attr('fill', '#fff')
            .attr('font-weight', '600')
            .text(function(d) { return d.count; })
            .attr('opacity', 0)
            .transition().duration(400).delay(function(d, i) { return 500 + i * 50; })
            .attr('opacity', 1);
    })
    .catch(function() {
        document.getElementById('domainBarContainer').innerHTML = '<div class="viz-empty">Could not load domain data.</div>';
    });
}

// ======================================================================
// Auto-load results on page load
// ======================================================================
loadResults();
</script>"""


# ---------------------------------------------------------------------------
# 404 page
# ---------------------------------------------------------------------------

PAGE_404 = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - Page Not Found</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            text-align: center;
        }
        .container { max-width: 480px; padding: 40px; }
        h1 { font-size: 72px; color: #00A651; margin: 0; }
        p { color: #8892a4; font-size: 16px; margin: 16px 0; }
        a {
            color: #00A651;
            text-decoration: none;
            font-weight: 600;
            border: 1px solid #00A651;
            padding: 10px 24px;
            border-radius: 6px;
            display: inline-block;
            margin-top: 16px;
            transition: all 0.15s;
        }
        a:hover { background: #00A651; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h1>404</h1>
        <p>The page you are looking for does not exist.</p>
        <a href="index.html">Back to Taxonomy Explorer</a>
    </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------


def build():
    print("=" * 60)
    print("  Building static site -> dist/")
    print("=" * 60)

    # Clean and create dist directory
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load source data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading source data...")

    extractions = load_json(OUTPUT_DIR / "extractions.json")
    taxonomy = load_json(OUTPUT_DIR / "taxonomy.json")
    crosslinks = load_json(OUTPUT_DIR / "crosslinks.json")
    dedup = load_json(OUTPUT_DIR / "deduplicated.json")
    graph_json = load_json(OUTPUT_DIR / "knowledge_graph.json")

    # ------------------------------------------------------------------
    # 2. Copy raw data files to dist/data/
    # ------------------------------------------------------------------
    print("\n[2/6] Copying data files...")

    data_files = {
        "taxonomy.json": OUTPUT_DIR / "taxonomy.json",
        "extractions.json": OUTPUT_DIR / "extractions.json",
        "crosslinks.json": OUTPUT_DIR / "crosslinks.json",
        "deduplicated.json": OUTPUT_DIR / "deduplicated.json",
    }
    for dest_name, src_path in data_files.items():
        if src_path.exists():
            shutil.copy2(src_path, DATA_DIR / dest_name)
            print(f"  -> data/{dest_name}")
        else:
            print(f"  [skip] {src_path.name} not found")

    # ------------------------------------------------------------------
    # 3. Pre-compute derived data files
    # ------------------------------------------------------------------
    print("\n[3/6] Pre-computing derived data...")

    # Collect all concepts/entities/relationships for graph-data
    concepts = dedup.get("concepts", []) if dedup else extractions.get("concepts", [])
    entities = dedup.get("entities", []) if dedup else extractions.get("entities", [])
    relationships = extractions.get("relationships", [])
    if crosslinks:
        relationships = relationships + crosslinks.get("new_relationships", [])

    # graph-data.json
    graph_data = compute_graph_data(concepts, entities, relationships)
    save_json(graph_data, DATA_DIR / "graph-data.json")

    # relationship-types.json
    rel_types = compute_relationship_types(relationships)
    save_json(rel_types, DATA_DIR / "relationship-types.json")

    # concept-data-points.json
    concept_dp = compute_concept_data_points(extractions)
    save_json(concept_dp, DATA_DIR / "concept-data-points.json")

    # stats.json
    stats = compute_stats(extractions, dedup, graph_json, crosslinks)

    # ------------------------------------------------------------------
    # 4. Copy export/download files
    # ------------------------------------------------------------------
    print("\n[4/6] Copying download files...")

    download_files = [
        "knowledge_graph.json",
        "knowledge_graph.graphml",
        "relationships.csv",
        "taxonomy_tree.md",
        "knowledge_graph.html",
        "summary_report.html",
        "knowledge_graph.md",
        "taxonomy.json",
        "extractions.json",
        "crosslinks.json",
    ]

    available_exports = []
    for fname in download_files:
        src = OUTPUT_DIR / fname
        if src.exists():
            shutil.copy2(src, DOWNLOADS_DIR / fname)
            # Find the label
            label = fname
            for name, lbl in EXPORT_LABELS:
                if name == fname:
                    label = lbl
                    break
            available_exports.append({"name": fname, "label": label})
            print(f"  -> downloads/{fname}")

    # Create obsidian vault zip if it exists
    vault_dir = OUTPUT_DIR / "obsidian_vault"
    if vault_dir.exists():
        zip_path = DOWNLOADS_DIR / "obsidian_vault.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in vault_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(vault_dir)
                    zf.write(file_path, arcname)
        available_exports.append(
            {"name": "obsidian_vault.zip", "label": "Obsidian Vault (Zip)"}
        )
        print(f"  -> downloads/obsidian_vault.zip")

    # Update stats with available exports and save
    stats["available_exports"] = available_exports
    save_json(stats, DATA_DIR / "stats.json")

    # ------------------------------------------------------------------
    # 5. Transform and write index.html
    # ------------------------------------------------------------------
    print("\n[5/6] Transforming index.html...")

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        original_html = f.read()

    static_html = transform_html(original_html)

    with open(DIST_DIR / "index.html", "w", encoding="utf-8") as f:
        f.write(static_html)
    print("  -> index.html")

    # ------------------------------------------------------------------
    # 6. Write 404 page
    # ------------------------------------------------------------------
    print("\n[6/6] Writing 404 page...")
    with open(DIST_DIR / "404.html", "w", encoding="utf-8") as f:
        f.write(PAGE_404)
    print("  -> 404.html")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Build complete!")
    print("=" * 60)

    # Count files
    total_files = sum(1 for _ in DIST_DIR.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in DIST_DIR.rglob("*") if f.is_file())
    print(f"  Total files: {total_files}")
    print(f"  Total size:  {total_size / 1024 / 1024:.1f} MB")
    print(f"  Output dir:  {DIST_DIR}")
    print()


if __name__ == "__main__":
    build()
