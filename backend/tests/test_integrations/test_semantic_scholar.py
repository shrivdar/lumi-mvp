"""Tests for Semantic Scholar tool."""

from __future__ import annotations

import httpx
import pytest
import respx

from integrations.semantic_scholar import SemanticScholarTool

S2_BASE = "https://api.semanticscholar.org/graph/v1"


SEARCH_RESPONSE = {
    "total": 1,
    "data": [
        {
            "paperId": "abc123",
            "title": "B7-H3 as a checkpoint in cancer immunotherapy",
            "abstract": "A review of B7-H3 checkpoint molecule.",
            "year": 2023,
            "citationCount": 42,
            "influentialCitationCount": 8,
            "tldr": {"text": "B7-H3 is an important immune checkpoint."},
            "externalIds": {"DOI": "10.1234/test", "PubMed": "99999"},
            "authors": [{"name": "Jane Doe"}, {"name": "John Smith"}],
            "journal": {"name": "Nature Immunology"},
            "fieldsOfStudy": ["Medicine", "Biology"],
        }
    ],
}


@pytest.fixture
def s2() -> SemanticScholarTool:
    return SemanticScholarTool()


class TestSemanticScholarSearch:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self, s2: SemanticScholarTool) -> None:
        respx.get(f"{S2_BASE}/paper/search").mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE)
        )
        result = await s2.execute(action="search", query="B7-H3 cancer")
        assert result["total"] == 1
        paper = result["papers"][0]
        assert paper["paper_id"] == "abc123"
        assert paper["doi"] == "10.1234/test"
        assert paper["pmid"] == "99999"
        assert "Jane Doe" in paper["authors"]
        assert paper["citation_count"] == 42
        assert paper["tldr"] == "B7-H3 is an important immune checkpoint."
        await s2.close()


class TestSemanticScholarPaper:
    @respx.mock
    @pytest.mark.asyncio
    async def test_get_paper(self, s2: SemanticScholarTool) -> None:
        paper_data = SEARCH_RESPONSE["data"][0].copy()
        paper_data["referenceCount"] = 15
        respx.get(f"{S2_BASE}/paper/abc123").mock(
            return_value=httpx.Response(200, json=paper_data)
        )
        result = await s2.execute(action="paper", paper_id="abc123")
        assert result["paper"]["title"] == "B7-H3 as a checkpoint in cancer immunotherapy"
        await s2.close()


class TestSemanticScholarCitations:
    @respx.mock
    @pytest.mark.asyncio
    async def test_citations(self, s2: SemanticScholarTool) -> None:
        respx.get(f"{S2_BASE}/paper/abc123/citations").mock(
            return_value=httpx.Response(200, json={
                "data": [
                    {"citingPaper": {
                        "paperId": "xyz", "title": "Citing paper",
                        "year": 2024, "citationCount": 5, "authors": [],
                    }},
                ]
            })
        )
        result = await s2.execute(action="citations", paper_id="abc123")
        assert len(result["citations"]) == 1
        assert result["citations"][0]["paper_id"] == "xyz"
        await s2.close()
