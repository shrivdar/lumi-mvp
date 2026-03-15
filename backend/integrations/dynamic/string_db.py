"""Auto-generated tool wrapper: string_db

STRING database tool
"""

import urllib.request
import json

def run(*, query: str = "", species: int = 9606, limit: int = 10, **kwargs) -> dict:
    """Query STRING-db for protein-protein interactions."""
    base = "https://string-db.org/api/json"
    params = f"identifiers={query}&species={species}&limit={limit}"
    url = f"{base}/network?{params}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        interactions = []
        for item in data:
            interactions.append({
                "protein_a": item.get("preferredName_A", ""),
                "protein_b": item.get("preferredName_B", ""),
                "score": item.get("score", 0),
            })
        return {"query": query, "interactions": interactions}
    except Exception as e:
        return {"error": str(e), "query": query, "interactions": []}
