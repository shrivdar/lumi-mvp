#!/usr/bin/env python3
"""Export a YOHAS Knowledge Graph to various formats.

Usage:
    python -m scripts.export_graph --input demo-kg.json --format json cytoscape markdown graphml
    python -m scripts.export_graph --input demo-kg.json --format cytoscape -o output/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from repo root without installing as package
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backend.world_model.knowledge_graph import InMemoryKnowledgeGraph  # noqa: E402
from backend.core.models import KGNode, KGEdge  # noqa: E402


def _load_kg(path: Path) -> InMemoryKnowledgeGraph:
    """Load a KG from a JSON file (to_json() format or seed_demo output)."""
    with open(path) as f:
        data = json.load(f)

    kg = InMemoryKnowledgeGraph()

    # Support both to_json() format and seed_demo.py format
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    for n in nodes:
        node = KGNode(**n) if isinstance(n, dict) else n
        kg.add_node(node)

    for e in edges:
        edge = KGEdge(**e) if isinstance(e, dict) else e
        kg.add_edge(edge)

    return kg


def _export_json(kg: InMemoryKnowledgeGraph, out_dir: Path) -> Path:
    """Export as raw JSON (nodes + edges + metadata)."""
    dest = out_dir / "kg_export.json"
    dest.write_text(json.dumps(kg.to_json(), indent=2, default=str))
    return dest


def _export_cytoscape(kg: InMemoryKnowledgeGraph, out_dir: Path) -> Path:
    """Export as Cytoscape.js-compatible JSON."""
    dest = out_dir / "kg_cytoscape.json"
    dest.write_text(json.dumps(kg.to_cytoscape(), indent=2, default=str))
    return dest


def _export_markdown(kg: InMemoryKnowledgeGraph, out_dir: Path) -> Path:
    """Export as Markdown summary."""
    dest = out_dir / "kg_summary.md"
    dest.write_text(kg.to_markdown_summary())
    return dest


def _export_graphml(kg: InMemoryKnowledgeGraph, out_dir: Path) -> Path:
    """Export as GraphML for Cytoscape Desktop / Gephi / yEd."""
    import xml.etree.ElementTree as ET

    graphml = ET.Element("graphml", xmlns="http://graphml.graphstruct.org/xmlns")

    # Attribute definitions
    for attr_id, attr_name, attr_for, attr_type in [
        ("d0", "label", "node", "string"),
        ("d1", "node_type", "node", "string"),
        ("d2", "confidence", "edge", "double"),
        ("d3", "relation", "edge", "string"),
        ("d4", "source_agent", "edge", "string"),
        ("d5", "is_falsified", "edge", "boolean"),
    ]:
        ET.SubElement(graphml, "key", id=attr_id, **{"for": attr_for, "attr.name": attr_name, "attr.type": attr_type})

    graph = ET.SubElement(graphml, "graph", id="KG", edgedefault="directed")

    data = kg.to_json()
    for node in data.get("nodes", []):
        n_el = ET.SubElement(graph, "node", id=node["id"])
        ET.SubElement(n_el, "data", key="d0").text = node.get("label", node["id"])
        ET.SubElement(n_el, "data", key="d1").text = node.get("node_type", "unknown")

    for edge in data.get("edges", []):
        e_el = ET.SubElement(graph, "edge", source=edge["source"], target=edge["target"], id=edge.get("id", ""))
        ET.SubElement(e_el, "data", key="d2").text = str(edge.get("confidence", 0.5))
        ET.SubElement(e_el, "data", key="d3").text = edge.get("relation", "RELATED_TO")
        ET.SubElement(e_el, "data", key="d4").text = edge.get("agent_id", "")
        ET.SubElement(e_el, "data", key="d5").text = str(edge.get("is_falsified", False)).lower()

    tree = ET.ElementTree(graphml)
    dest = out_dir / "kg_export.graphml"
    tree.write(str(dest), encoding="unicode", xml_declaration=True)
    return dest


EXPORTERS = {
    "json": _export_json,
    "cytoscape": _export_cytoscape,
    "markdown": _export_markdown,
    "graphml": _export_graphml,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOHAS Knowledge Graph")
    parser.add_argument("--input", "-i", required=True, help="Path to KG JSON file")
    parser.add_argument(
        "--format", "-f",
        nargs="+",
        choices=list(EXPORTERS.keys()),
        default=["json"],
        help="Export format(s)",
    )
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading KG from {input_path}...")
    kg = _load_kg(input_path)
    print(f"  {len(kg._nodes)} nodes, {len(kg._edges)} edges")

    for fmt in args.format:
        dest = EXPORTERS[fmt](kg, out_dir)
        print(f"  [{fmt}] -> {dest}")

    print("Done.")


if __name__ == "__main__":
    main()
