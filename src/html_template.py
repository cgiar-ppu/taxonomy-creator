"""HTML template for interactive knowledge graph visualization using vis.js."""

GRAPH_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph - Taxonomy Creator</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            overflow: hidden;
        }}
        #header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 56px;
            background: #16213e;
            border-bottom: 2px solid #0f3460;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            z-index: 1000;
        }}
        #header h1 {{
            font-size: 18px;
            font-weight: 600;
            color: #e94560;
        }}
        #header .stats {{
            font-size: 13px;
            color: #999;
        }}
        #search-container {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        #search-input {{
            width: 260px;
            padding: 6px 12px;
            border: 1px solid #0f3460;
            border-radius: 4px;
            background: #1a1a2e;
            color: #eee;
            font-size: 13px;
            outline: none;
        }}
        #search-input:focus {{
            border-color: #e94560;
        }}
        #search-input::placeholder {{
            color: #666;
        }}
        #search-results {{
            position: fixed;
            top: 56px;
            right: 20px;
            width: 280px;
            max-height: 300px;
            overflow-y: auto;
            background: #16213e;
            border: 1px solid #0f3460;
            border-radius: 0 0 4px 4px;
            z-index: 999;
            display: none;
        }}
        #search-results .result-item {{
            padding: 8px 12px;
            cursor: pointer;
            border-bottom: 1px solid #0f3460;
            font-size: 13px;
        }}
        #search-results .result-item:hover {{
            background: #0f3460;
        }}
        #search-results .result-item .type-badge {{
            display: inline-block;
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 10px;
            margin-left: 6px;
            color: #fff;
        }}
        #graph-container {{
            position: fixed;
            top: 56px;
            left: 0;
            right: 0;
            bottom: 0;
        }}
        #legend {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(22, 33, 62, 0.95);
            border: 1px solid #0f3460;
            border-radius: 6px;
            padding: 12px 16px;
            z-index: 1000;
            max-height: 400px;
            overflow-y: auto;
            font-size: 12px;
        }}
        #legend h3 {{
            font-size: 13px;
            margin-bottom: 8px;
            color: #e94560;
        }}
        #legend .section-title {{
            font-size: 11px;
            color: #999;
            margin-top: 8px;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 3px 0;
        }}
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 8px;
            flex-shrink: 0;
        }}
        .legend-line {{
            width: 20px;
            height: 3px;
            margin-right: 8px;
            flex-shrink: 0;
            border-radius: 1px;
        }}
        #node-info {{
            position: fixed;
            top: 70px;
            right: 20px;
            width: 300px;
            background: rgba(22, 33, 62, 0.95);
            border: 1px solid #0f3460;
            border-radius: 6px;
            padding: 16px;
            z-index: 1000;
            display: none;
            font-size: 13px;
        }}
        #node-info h3 {{
            color: #e94560;
            margin-bottom: 8px;
        }}
        #node-info .info-row {{
            margin: 4px 0;
        }}
        #node-info .info-label {{
            color: #999;
            font-size: 11px;
            text-transform: uppercase;
        }}
        #node-info .connections {{
            margin-top: 10px;
            max-height: 200px;
            overflow-y: auto;
        }}
        #node-info .conn-item {{
            padding: 3px 0;
            font-size: 12px;
            color: #ccc;
        }}
        #toggle-legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #16213e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 8px 14px;
            border-radius: 4px;
            cursor: pointer;
            z-index: 1000;
            font-size: 12px;
        }}
        #toggle-legend:hover {{
            background: #0f3460;
        }}
        #controls {{
            position: fixed;
            top: 70px;
            left: 20px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            z-index: 1000;
        }}
        #controls button {{
            background: #16213e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        #controls button:hover {{
            background: #0f3460;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Knowledge Graph - Taxonomy Creator</h1>
        <div class="stats">{total_nodes} nodes | {total_edges} edges</div>
        <div id="search-container">
            <input type="text" id="search-input" placeholder="Search nodes..." autocomplete="off">
        </div>
    </div>

    <div id="search-results"></div>

    <div id="controls">
        <button onclick="resetView()">Reset View</button>
        <button onclick="togglePhysics()">Toggle Physics</button>
    </div>

    <div id="graph-container"></div>

    <div id="legend">
        <h3>Legend</h3>
        <div class="section-title">Node Types</div>
        <div id="node-legend"></div>
        <div class="section-title">Edge Types</div>
        <div id="edge-legend"></div>
    </div>

    <div id="node-info">
        <h3 id="info-title">Node</h3>
        <div id="info-content"></div>
    </div>

    <button id="toggle-legend" onclick="toggleLegend()">Toggle Legend</button>

    <script>
        // Data injected from Python
        var nodesData = {nodes_json};
        var edgesData = {edges_json};
        var legendData = {legend_json};

        // Create vis.js DataSets
        var nodes = new vis.DataSet(nodesData);
        var edges = new vis.DataSet(edgesData);

        // Graph options
        var options = {{
            nodes: {{
                shape: 'dot',
                borderWidth: 1,
                borderWidthSelected: 3,
                shadow: true,
                font: {{
                    color: '#eee',
                    strokeWidth: 3,
                    strokeColor: '#1a1a2e',
                }},
            }},
            edges: {{
                smooth: {{
                    type: 'continuous',
                    roundness: 0.3,
                }},
                shadow: false,
                font: {{
                    color: '#999',
                    strokeWidth: 0,
                    size: 8,
                }},
            }},
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -80,
                    centralGravity: 0.01,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.4,
                    avoidOverlap: 0.3,
                }},
                stabilization: {{
                    iterations: 200,
                    fit: true,
                }},
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: true,
                hideEdgesOnZoom: true,
                multiselect: true,
            }},
            layout: {{
                improvedLayout: true,
            }},
        }};

        // Create the network
        var container = document.getElementById('graph-container');
        var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);

        var physicsEnabled = true;

        // Build legend
        function buildLegend() {{
            var nodeLegend = document.getElementById('node-legend');
            var edgeLegend = document.getElementById('edge-legend');

            legendData.forEach(function(item) {{
                var div = document.createElement('div');
                div.className = 'legend-item';
                if (item.type === 'node') {{
                    div.innerHTML = '<div class="legend-color" style="background:' + item.color + '"></div>' + item.label;
                    nodeLegend.appendChild(div);
                }} else {{
                    div.innerHTML = '<div class="legend-line" style="background:' + item.color + '"></div>' + item.label;
                    edgeLegend.appendChild(div);
                }}
            }});
        }}
        buildLegend();

        // Search functionality
        var searchInput = document.getElementById('search-input');
        var searchResults = document.getElementById('search-results');

        searchInput.addEventListener('input', function() {{
            var query = this.value.toLowerCase().trim();
            searchResults.innerHTML = '';

            if (query.length < 2) {{
                searchResults.style.display = 'none';
                // Reset highlighting
                nodes.forEach(function(node) {{
                    nodes.update({{ id: node.id, opacity: 1 }});
                }});
                return;
            }}

            var matches = nodesData.filter(function(n) {{
                return n.label.toLowerCase().includes(query);
            }}).slice(0, 15);

            if (matches.length > 0) {{
                searchResults.style.display = 'block';
                matches.forEach(function(m) {{
                    var div = document.createElement('div');
                    div.className = 'result-item';
                    var badgeColor = m.color || '#999';
                    div.innerHTML = m.label + '<span class="type-badge" style="background:' + badgeColor + '">' + (m.group || 'node') + '</span>';
                    div.addEventListener('click', function() {{
                        focusNode(m.id);
                        searchResults.style.display = 'none';
                    }});
                    searchResults.appendChild(div);
                }});
            }} else {{
                searchResults.style.display = 'none';
            }}
        }});

        function focusNode(nodeId) {{
            network.focus(nodeId, {{
                scale: 1.5,
                animation: {{ duration: 500, easingFunction: 'easeInOutQuad' }},
            }});
            network.selectNodes([nodeId]);
            showNodeInfo(nodeId);
        }}

        function showNodeInfo(nodeId) {{
            var node = nodes.get(nodeId);
            if (!node) return;

            var infoDiv = document.getElementById('node-info');
            var titleDiv = document.getElementById('info-title');
            var contentDiv = document.getElementById('info-content');

            titleDiv.textContent = node.label;

            var html = '';
            html += '<div class="info-row"><span class="info-label">Type:</span> ' + (node.group || 'unknown') + '</div>';
            html += '<div class="info-row"><span class="info-label">Frequency:</span> ' + (node.size ? Math.round((node.size - 10) / 2) : 1) + '</div>';

            // Find connections
            var connEdges = network.getConnectedEdges(nodeId);
            var connNodes = network.getConnectedNodes(nodeId);

            html += '<div class="info-row"><span class="info-label">Connections:</span> ' + connNodes.length + '</div>';

            if (connNodes.length > 0) {{
                html += '<div class="connections"><span class="info-label">Connected to:</span>';
                connNodes.forEach(function(cn) {{
                    var connNode = nodes.get(cn);
                    if (connNode) {{
                        html += '<div class="conn-item">- ' + connNode.label + '</div>';
                    }}
                }});
                html += '</div>';
            }}

            contentDiv.innerHTML = html;
            infoDiv.style.display = 'block';
        }}

        // Click handler
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                showNodeInfo(params.nodes[0]);
            }} else {{
                document.getElementById('node-info').style.display = 'none';
            }}
        }});

        // Controls
        function resetView() {{
            network.fit({{ animation: {{ duration: 500 }} }});
            document.getElementById('node-info').style.display = 'none';
        }}

        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}

        function toggleLegend() {{
            var legend = document.getElementById('legend');
            legend.style.display = legend.style.display === 'none' ? 'block' : 'none';
        }}

        // Close search results when clicking elsewhere
        document.addEventListener('click', function(e) {{
            if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {{
                searchResults.style.display = 'none';
            }}
        }});
    </script>
</body>
</html>"""
