"""PubMed / NCBI E-utilities client — literature search and article retrieval."""

from __future__ import annotations

from typing import Any
from xml.etree import ElementTree

from core.config import settings
from core.constants import CACHE_TTL_PUBMED, RATE_LIMIT_PUBMED
from integrations.base_tool import BaseTool

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedTool(BaseTool):
    tool_id = "pubmed"
    name = "pubmed_search"
    description = (
        "Search PubMed for biomedical literature and retrieve article metadata with abstracts and DOIs."
    )
    category = "literature"
    rate_limit = RATE_LIMIT_PUBMED
    cache_ttl = CACHE_TTL_PUBMED

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search")
        if action == "search":
            return await self._search(
                query=kwargs["query"],
                max_results=kwargs.get("max_results", 20),
                sort=kwargs.get("sort", "relevance"),
            )
        elif action == "fetch":
            return await self._fetch(pmids=kwargs["pmids"])
        elif action == "cited_by":
            return await self._cited_by(pmid=kwargs["pmid"])
        raise ValueError(f"Unknown PubMed action: {action}")

    # ------------------------------------------------------------------

    async def _search(self, query: str, max_results: int = 20, sort: str = "relevance") -> dict[str, Any]:
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 100),
            "retmode": "json",
            "sort": sort,
        }
        if settings.ncbi_api_key:
            params["api_key"] = settings.ncbi_api_key

        resp = await self._http.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return {"articles": [], "total_count": 0, "query": query}

        return await self._fetch(pmids=id_list)

    async def _fetch(self, pmids: list[str]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        if settings.ncbi_api_key:
            params["api_key"] = settings.ncbi_api_key

        resp = await self._http.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
        resp.raise_for_status()

        articles = self._parse_xml(resp.text)
        return {"articles": articles, "total_count": len(articles), "pmids": pmids}

    async def _cited_by(self, pmid: str) -> dict[str, Any]:
        params: dict[str, Any] = {
            "dbfrom": "pubmed",
            "db": "pubmed",
            "id": pmid,
            "linkname": "pubmed_pubmed_citedin",
            "retmode": "json",
        }
        if settings.ncbi_api_key:
            params["api_key"] = settings.ncbi_api_key

        resp = await self._http.get(f"{EUTILS_BASE}/elink.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        links = data.get("linksets", [{}])[0].get("linksetdbs", [{}])
        citing_ids: list[str] = []
        for db in links:
            citing_ids.extend(str(lid.get("id", "")) for lid in db.get("links", []))
        return {"pmid": pmid, "cited_by_count": len(citing_ids), "citing_pmids": citing_ids}

    # ------------------------------------------------------------------
    # XML parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xml(xml_text: str) -> list[dict[str, Any]]:
        articles: list[dict[str, Any]] = []
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return articles

        for art_elem in root.findall(".//PubmedArticle"):
            article: dict[str, Any] = {}
            medline = art_elem.find(".//MedlineCitation")
            if medline is None:
                continue

            pmid_el = medline.find("PMID")
            article["pmid"] = pmid_el.text if pmid_el is not None else ""

            art_info = medline.find("Article")
            if art_info is not None:
                title_el = art_info.find("ArticleTitle")
                article["title"] = title_el.text if title_el is not None else ""

                abstract_el = art_info.find("Abstract/AbstractText")
                article["abstract"] = abstract_el.text if abstract_el is not None else ""

                # Authors
                authors = []
                for author in art_info.findall(".//Author"):
                    last = author.findtext("LastName", "")
                    first = author.findtext("ForeName", "")
                    if last:
                        authors.append(f"{last} {first}".strip())
                article["authors"] = authors

                # Journal
                journal = art_info.find("Journal")
                if journal is not None:
                    article["journal"] = journal.findtext("Title", "")
                    pub_date = journal.find("JournalIssue/PubDate")
                    if pub_date is not None:
                        year = pub_date.findtext("Year", "")
                        article["year"] = int(year) if year.isdigit() else None

                # DOI
                for eid in art_info.findall(".//ELocationID"):
                    if eid.get("EIdType") == "doi":
                        article["doi"] = eid.text

            # MeSH terms
            mesh_terms = []
            for mesh in medline.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)
            article["mesh_terms"] = mesh_terms

            articles.append(article)
        return articles
