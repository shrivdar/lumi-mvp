# NetworkX Recipes

## Purpose

Common NetworkX patterns for biological network analysis — protein interactions, pathway graphs, gene regulatory networks, and knowledge graph operations.

## Recipes

### Build Protein Interaction Network

```python
import networkx as nx

# Build from edge list with confidence scores
G = nx.Graph()

interactions = [
    ("BRAF", "KRAS", {"confidence": 0.95, "source": "STRING"}),
    ("BRAF", "MAP2K1", {"confidence": 0.99, "source": "STRING"}),
    ("MAP2K1", "MAPK1", {"confidence": 0.98, "source": "STRING"}),
    ("MAPK1", "ELK1", {"confidence": 0.85, "source": "STRING"}),
    ("KRAS", "RAF1", {"confidence": 0.92, "source": "STRING"}),
]
G.add_edges_from(interactions)

# Add node attributes
node_attrs = {"BRAF": {"type": "kinase", "druggable": True},
              "KRAS": {"type": "GTPase", "druggable": True}}
nx.set_node_attributes(G, node_attrs)

# Filter by confidence
high_conf = nx.Graph([(u, v, d) for u, v, d in G.edges(data=True) if d["confidence"] > 0.9])
```

### Centrality Analysis for Target Identification

```python
# Compute centrality metrics
degree = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

# Combine into a ranking DataFrame
import pandas as pd
centrality_df = pd.DataFrame({
    "degree": degree,
    "betweenness": betweenness,
    "closeness": closeness,
    "eigenvector": eigenvector,
})

# Rank by composite score
centrality_df["composite"] = (
    centrality_df["degree"].rank(pct=True) * 0.25 +
    centrality_df["betweenness"].rank(pct=True) * 0.35 +
    centrality_df["closeness"].rank(pct=True) * 0.15 +
    centrality_df["eigenvector"].rank(pct=True) * 0.25
)
top_targets = centrality_df.sort_values("composite", ascending=False).head(10)
```

### Community Detection for Pathway Modules

```python
from networkx.algorithms.community import louvain_communities, modularity

# Detect communities (pathway modules)
communities = louvain_communities(G, weight="confidence", seed=42)

# Calculate modularity
mod_score = modularity(G, communities, weight="confidence")
print(f"Modularity: {mod_score:.3f}")  # > 0.3 indicates meaningful structure

# Label nodes with community membership
for i, comm in enumerate(communities):
    for node in comm:
        G.nodes[node]["community"] = i

# Find inter-community edges (pathway cross-talk)
cross_talk = [(u, v) for u, v in G.edges()
              if G.nodes[u]["community"] != G.nodes[v]["community"]]
```

### Shortest Path and Disease-Target Distance

```python
# Find shortest paths between disease genes and drug targets
disease_genes = ["TP53", "BRCA1", "APC"]
drug_targets = ["EGFR", "BRAF", "VEGFR2"]

for dg in disease_genes:
    for dt in drug_targets:
        if nx.has_path(G, dg, dt):
            path = nx.shortest_path(G, dg, dt, weight=None)
            path_length = len(path) - 1
            print(f"{dg} -> {dt}: {' -> '.join(path)} (distance: {path_length})")

# Network proximity score (average shortest path between two gene sets)
def network_proximity(G, set_a, set_b):
    """Average shortest path between two gene sets."""
    distances = []
    for a in set_a:
        for b in set_b:
            if a in G and b in G and nx.has_path(G, a, b):
                distances.append(nx.shortest_path_length(G, a, b))
    return sum(distances) / len(distances) if distances else float("inf")
```

### Directed Pathway Graph (Signaling Cascade)

```python
# Directed graph for signaling pathways
DG = nx.DiGraph()

# Add regulatory edges
DG.add_edge("EGF", "EGFR", relation="activates")
DG.add_edge("EGFR", "KRAS", relation="activates")
DG.add_edge("KRAS", "BRAF", relation="activates")
DG.add_edge("BRAF", "MEK1", relation="phosphorylates")
DG.add_edge("MEK1", "ERK1", relation="phosphorylates")
DG.add_edge("ERK1", "MYC", relation="activates")

# Find all upstream regulators of a target
def get_upstream(DG, target, max_depth=5):
    """Get all nodes upstream of target within max_depth."""
    upstream = set()
    for source in DG.nodes():
        if source != target and nx.has_path(DG, source, target):
            path_len = nx.shortest_path_length(DG, source, target)
            if path_len <= max_depth:
                upstream.add((source, path_len))
    return sorted(upstream, key=lambda x: x[1])

# Find all downstream effectors
def get_downstream(DG, source, max_depth=5):
    """Get all nodes downstream of source within max_depth."""
    return [(t, nx.shortest_path_length(DG, source, t))
            for t in nx.descendants(DG, source)
            if nx.shortest_path_length(DG, source, t) <= max_depth]

# Identify feedback loops
cycles = list(nx.simple_cycles(DG))
```

### Knowledge Graph Subgraph Extraction

```python
# Extract subgraph around a seed node (like KG context building)
def extract_subgraph(G, seed_node, hops=2):
    """Extract neighborhood subgraph within N hops of seed node."""
    nodes = {seed_node}
    frontier = {seed_node}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(G.neighbors(node))
        nodes.update(next_frontier)
        frontier = next_frontier
    return G.subgraph(nodes).copy()

# Merge two subgraphs
def merge_subgraphs(g1, g2):
    """Merge two subgraphs, combining edge attributes."""
    merged = nx.compose(g1, g2)
    return merged
```

### Export for Visualization

```python
# Export to Cytoscape-compatible JSON
import json

def to_cytoscape_json(G):
    """Convert NetworkX graph to Cytoscape.js JSON format."""
    elements = {"nodes": [], "edges": []}
    for node, data in G.nodes(data=True):
        elements["nodes"].append({"data": {"id": node, **data}})
    for u, v, data in G.edges(data=True):
        elements["edges"].append({"data": {"source": u, "target": v, **data}})
    return elements

cyto_json = to_cytoscape_json(G)
with open("network.json", "w") as f:
    json.dump(cyto_json, f, indent=2, default=str)
```

## Common Pitfalls

- **Graph type matters**: Use `DiGraph` for directed relationships (signaling, regulation), `Graph` for undirected (physical interaction, co-expression). Mixing them silently drops directionality.
- **Disconnected components**: Many biological networks have disconnected components. `shortest_path` raises `NetworkXNoPath` for unreachable nodes — always check `has_path` first.
- **Weighted vs. unweighted**: Centrality metrics change significantly with weights. Ensure weights represent distance (lower = closer) not similarity (higher = closer). Invert similarity to distance if needed.
- **Scalability**: NetworkX is memory-heavy for >100k nodes. Use graph-tool or igraph for large networks.
- **Self-loops**: Some databases include self-interactions. Remove with `G.remove_edges_from(nx.selfloop_edges(G))` if not meaningful.
