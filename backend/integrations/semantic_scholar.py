"""Semantic Scholar API client — paper search, citation graphs, author info."""

from __future__ import annotations

from typing import Any

from core.config import settings
from core.constants import CACHE_TTL_SEMANTIC_SCHOLAR, RATE_LIMIT_SEMANTIC_SCHOLAR
from integrations.base_tool import BaseTool

S2_BASE = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarTool(BaseTool):
    tool_id = "semantic_scholar"
    name = "semantic_scholar_search"
    description = (
        "Search Semantic Scholar for academic papers with citation graphs, TLDRs, and influence scores."
    )
    category = "literature"
    rate_limit = RATE_LIMIT_SEMANTIC_SCHOLAR
    cache_ttl = CACHE_TTL_SEMANTIC_SCHOLAR

    @property
    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if settings.s2_api_key:
            h["x-api-key"] = settings.s2_api_key
        return h

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(
                query=kwargs["query"],
                max_results=kwargs.get("max_results", 20),
                fields=kwargs.get("fields"),
            )
        elif action == "paper":
            return await self._get_paper(paper_id=kwargs["paper_id"])
        elif action == "citations":
            return await self._citations(paper_id=kwargs["paper_id"], max_results=kwargs.get("max_results", 50))
        elif action == "references":
            return await self._references(paper_id=kwargs["paper_id"], max_results=kwargs.get("max_results", 50))
        raise ValueError(f"Unknown Semantic Scholar action: {action}")

    async def _search(
        self, query: str, max_results: int = 20, fields: list[str] | None = None,
    ) -> dict[str, Any]:
        default_fields = [
            "paperId", "title", "abstract", "year", "citationCount",
            "influentialCitationCount", "tldr", "externalIds", "authors",
            "journal", "fieldsOfStudy",
        ]
        resp = await self._http.get(
            f"{S2_BASE}/paper/search",
            params={
                "query": query,
                "limit": min(max_results, 100),
                "fields": ",".join(fields or default_fields),
            },
            headers=self._headers,
        )
        resp.raise_for_status()
        data = resp.json()
        papers = [self._normalize(p) for p in data.get("data", [])]
        return {"papers": papers, "total": data.get("total", len(papers)), "query": query}

    async def _get_paper(self, paper_id: str) -> dict[str, Any]:
        fields = (
            "paperId,title,abstract,year,citationCount,influentialCitationCount,"
            "tldr,externalIds,authors,journal,fieldsOfStudy,referenceCount"
        )
        resp = await self._http.get(
            f"{S2_BASE}/paper/{paper_id}",
            params={"fields": fields},
            headers=self._headers,
        )
        resp.raise_for_status()
        return {"paper": self._normalize(resp.json())}

    async def _citations(self, paper_id: str, max_results: int = 50) -> dict[str, Any]:
        resp = await self._http.get(
            f"{S2_BASE}/paper/{paper_id}/citations",
            params={"fields": "paperId,title,year,citationCount,authors", "limit": min(max_results, 100)},
            headers=self._headers,
        )
        resp.raise_for_status()
        data = resp.json()
        citing = [self._normalize(c.get("citingPaper", {})) for c in data.get("data", [])]
        return {"paper_id": paper_id, "citations": citing, "total": len(citing)}

    async def _references(self, paper_id: str, max_results: int = 50) -> dict[str, Any]:
        resp = await self._http.get(
            f"{S2_BASE}/paper/{paper_id}/references",
            params={"fields": "paperId,title,year,citationCount,authors", "limit": min(max_results, 100)},
            headers=self._headers,
        )
        resp.raise_for_status()
        data = resp.json()
        refs = [self._normalize(r.get("citedPaper", {})) for r in data.get("data", [])]
        return {"paper_id": paper_id, "references": refs, "total": len(refs)}

    @staticmethod
    def _normalize(paper: dict[str, Any]) -> dict[str, Any]:
        ext_ids = paper.get("externalIds") or {}
        authors_raw = paper.get("authors") or []
        return {
            "paper_id": paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "year": paper.get("year"),
            "citation_count": paper.get("citationCount", 0),
            "influential_citation_count": paper.get("influentialCitationCount", 0),
            "tldr": (paper.get("tldr") or {}).get("text", ""),
            "doi": ext_ids.get("DOI", ""),
            "pmid": ext_ids.get("PubMed", ""),
            "arxiv_id": ext_ids.get("ArXiv", ""),
            "authors": [a.get("name", "") for a in authors_raw],
            "journal": (paper.get("journal") or {}).get("name", ""),
            "fields_of_study": paper.get("fieldsOfStudy") or [],
        }
