"""Tests for PubMed tool — search and fetch with mocked HTTP."""

from __future__ import annotations

import httpx
import pytest
import respx

from integrations.pubmed import PubMedTool

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

SEARCH_RESPONSE = {
    "esearchresult": {
        "count": "2",
        "retmax": "2",
        "idlist": ["12345678", "87654321"],
    }
}

FETCH_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>B7-H3 in NSCLC: a comprehensive review</ArticleTitle>
        <Abstract>
          <AbstractText>B7-H3 is overexpressed in non-small cell lung cancer.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
        </AuthorList>
        <Journal>
          <Title>Journal of Thoracic Oncology</Title>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
        <ELocationID EIdType="doi">10.1234/jto.2024.001</ELocationID>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Carcinoma, Non-Small-Cell Lung</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


@pytest.fixture
def pubmed() -> PubMedTool:
    return PubMedTool()


class TestPubMedSearch:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search_returns_articles(self, pubmed: PubMedTool) -> None:
        # Mock esearch
        respx.get(f"{EUTILS_BASE}/esearch.fcgi").mock(
            return_value=httpx.Response(200, json=SEARCH_RESPONSE)
        )
        # Mock efetch
        respx.get(f"{EUTILS_BASE}/efetch.fcgi").mock(
            return_value=httpx.Response(200, text=FETCH_XML)
        )

        result = await pubmed.execute(action="search", query="B7-H3 NSCLC")
        assert len(result["articles"]) == 1
        article = result["articles"][0]
        assert article["pmid"] == "12345678"
        assert "B7-H3" in article["title"]
        assert article["doi"] == "10.1234/jto.2024.001"
        assert article["year"] == 2024
        assert "Smith John" in article["authors"]
        assert "Carcinoma, Non-Small-Cell Lung" in article["mesh_terms"]
        await pubmed.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_search(self, pubmed: PubMedTool) -> None:
        respx.get(f"{EUTILS_BASE}/esearch.fcgi").mock(
            return_value=httpx.Response(200, json={"esearchresult": {"idlist": []}})
        )
        result = await pubmed.execute(action="search", query="nonexistent12345")
        assert result["articles"] == []
        assert result["total_count"] == 0
        await pubmed.close()


class TestPubMedFetch:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fetch_by_pmids(self, pubmed: PubMedTool) -> None:
        respx.get(f"{EUTILS_BASE}/efetch.fcgi").mock(
            return_value=httpx.Response(200, text=FETCH_XML)
        )
        result = await pubmed.execute(action="fetch", pmids=["12345678"])
        assert len(result["articles"]) == 1
        assert result["articles"][0]["abstract"] == "B7-H3 is overexpressed in non-small cell lung cancer."
        await pubmed.close()


class TestPubMedXMLParsing:
    def test_parse_malformed_xml(self) -> None:
        articles = PubMedTool._parse_xml("<bad xml")
        assert articles == []

    def test_parse_empty_article(self) -> None:
        xml = "<PubmedArticleSet><PubmedArticle><MedlineCitation></MedlineCitation></PubmedArticle></PubmedArticleSet>"
        articles = PubMedTool._parse_xml(xml)
        # MedlineCitation present but no Article child → still parsed with empty fields
        assert len(articles) == 1
        assert articles[0]["pmid"] == ""
