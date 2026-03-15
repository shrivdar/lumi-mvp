"""Tests for new tool integrations (tool expansion — 10 new tools)."""

from __future__ import annotations

import httpx
import pytest
import respx

from integrations.biogrid import BioGRIDTool
from integrations.cellxgene import CellxGeneTool
from integrations.clinvar import ClinVarTool
from integrations.depmap import DepMapTool
from integrations.gnomad import GnomADTool
from integrations.gtex import GTExTool
from integrations.hpo import HPOTool
from integrations.omim import OMIMTool
from integrations.opentargets import OpenTargetsTool
from integrations.string_db import StringDBTool

# ── OpenTargets ────────────────────────────────────

class TestOpenTargets:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self) -> None:
        tool = OpenTargetsTool()
        respx.get("https://api.platform.opentargets.org/api/v4/search").mock(
            return_value=httpx.Response(200, json={
                "data": [
                    {
                        "id": "ENSG00000141510", "name": "TP53",
                        "entity": "target", "description": "Tumor protein p53", "score": 1.0,
                    }
                ],
                "total": 1,
            })
        )
        result = await tool.execute(action="search", query="TP53")
        assert result["count"] == 1
        assert result["results"][0]["id"] == "ENSG00000141510"
        assert result["results"][0]["name"] == "TP53"
        await tool.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_associations(self) -> None:
        tool = OpenTargetsTool()
        respx.get("https://api.platform.opentargets.org/api/v4/association/filter").mock(
            return_value=httpx.Response(200, json={
                "data": [{
                    "targetId": "ENSG00000141510",
                    "diseaseId": "EFO_0000616",
                    "target": {"id": "ENSG00000141510", "approvedSymbol": "TP53"},
                    "disease": {"id": "EFO_0000616", "name": "neoplasm"},
                    "score": 0.95,
                    "datatypeScores": {},
                }],
                "total": 1,
            })
        )
        result = await tool.execute(action="associations", target_id="ENSG00000141510")
        assert result["count"] == 1
        assert result["associations"][0]["target_id"] == "ENSG00000141510"
        assert result["associations"][0]["score"] == 0.95
        await tool.close()


# ── ClinVar ────────────────────────────────────────

class TestClinVar:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self) -> None:
        tool = ClinVarTool()
        respx.get(url__regex=r".*esearch\.fcgi.*").mock(
            return_value=httpx.Response(200, json={
                "esearchresult": {"idlist": ["12345"], "count": "1"},
            })
        )
        respx.get(url__regex=r".*esummary\.fcgi.*").mock(
            return_value=httpx.Response(200, json={
                "result": {
                    "uids": ["12345"],
                    "12345": {
                        "title": "NM_000546.6(TP53):c.743G>A (p.Arg248Gln)",
                        "obj_type": "single nucleotide variant",
                        "clinical_significance": {
                            "description": "Pathogenic",
                            "review_status": "criteria provided, multiple submitters",
                            "last_evaluated": "2024-01-15",
                        },
                        "genes": [{"symbol": "TP53", "geneid": 7157}],
                        "accession": "VCV000012345",
                        "trait_set": [{"trait_name": "Li-Fraumeni syndrome"}],
                    },
                },
            })
        )
        result = await tool.execute(action="search", query="TP53 pathogenic")
        assert result["count"] == 1
        v = result["variants"][0]
        assert v["clinical_significance"] == "Pathogenic"
        assert v["genes"][0]["symbol"] == "TP53"
        assert v["accession"] == "VCV000012345"
        assert v["conditions"] == ["Li-Fraumeni syndrome"]
        await tool.close()


# ── GTEx ───────────────────────────────────────────

class TestGTEx:
    @respx.mock
    @pytest.mark.asyncio
    async def test_expression(self) -> None:
        tool = GTExTool()
        respx.get("https://gtexportal.org/api/v2/expression/medianGeneExpression").mock(
            return_value=httpx.Response(200, json={
                "data": [
                    {"tissueSiteDetailId": "Brain_Cortex", "tissueSiteDetail": "Brain - Cortex", "median": 15.5},
                    {"tissueSiteDetailId": "Liver", "tissueSiteDetail": "Liver", "median": 2.3},
                ],
            })
        )
        result = await tool.execute(action="expression", gene_id="ENSG00000141510.17")
        assert result["count"] == 2
        # Should be sorted by expression descending
        assert result["tissues"][0]["tissue_name"] == "Brain - Cortex"
        assert result["tissues"][0]["median_tpm"] == 15.5
        await tool.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_gene_search(self) -> None:
        tool = GTExTool()
        respx.get("https://gtexportal.org/api/v2/reference/gene").mock(
            return_value=httpx.Response(200, json={
                "data": [{
                    "gencodeId": "ENSG00000141510.17",
                    "geneSymbol": "TP53",
                    "description": "tumor protein p53",
                    "geneType": "protein_coding",
                    "chromosome": "chr17",
                }],
            })
        )
        result = await tool.execute(action="gene_search", query="TP53")
        assert result["count"] == 1
        assert result["genes"][0]["gene_symbol"] == "TP53"
        await tool.close()


# ── gnomAD ─────────────────────────────────────────

class TestGnomAD:
    @respx.mock
    @pytest.mark.asyncio
    async def test_variant(self) -> None:
        tool = GnomADTool()
        respx.post("https://gnomad.broadinstitute.org/api").mock(
            return_value=httpx.Response(200, json={
                "data": {
                    "variant": {
                        "variant_id": "17-7674220-G-A",
                        "chrom": "17",
                        "pos": 7674220,
                        "ref": "G",
                        "alt": "A",
                        "rsids": ["rs28934578"],
                        "exome": {"ac": 5, "an": 250000, "af": 0.00002, "populations": [], "filters": []},
                        "genome": {"ac": 2, "an": 150000, "af": 0.000013, "populations": [], "filters": []},
                        "clinvar_allele_id": "12345",
                        "transcript_consequences": [{
                            "gene_symbol": "TP53",
                            "gene_id": "ENSG00000141510",
                            "transcript_id": "ENST00000269305",
                            "hgvsc": "c.743G>A",
                            "hgvsp": "p.Arg248Gln",
                            "consequence_terms": ["missense_variant"],
                            "lof": "",
                            "polyphen_prediction": "probably_damaging",
                            "sift_prediction": "deleterious",
                        }],
                    },
                },
            })
        )
        result = await tool.execute(action="variant", variant_id="17-7674220-G-A")
        v = result["variant"]
        assert v["variant_id"] == "17-7674220-G-A"
        assert v["exome_af"] == 0.00002
        assert v["gene_symbol"] == "TP53"
        assert v["hgvsp"] == "p.Arg248Gln"
        await tool.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_gene_constraint(self) -> None:
        tool = GnomADTool()
        respx.post("https://gnomad.broadinstitute.org/api").mock(
            return_value=httpx.Response(200, json={
                "data": {
                    "gene": {
                        "gene_id": "ENSG00000141510",
                        "symbol": "TP53",
                        "name": "tumor protein p53",
                        "gnomad_constraint": {
                            "pLI": 0.999,
                            "oe_lof": 0.12,
                            "oe_lof_lower": 0.08,
                            "oe_lof_upper": 0.18,
                            "lof_z": 5.2,
                            "oe_mis": 0.65,
                            "mis_z": 3.1,
                            "oe_syn": 0.98,
                            "syn_z": -0.1,
                        },
                    },
                },
            })
        )
        result = await tool.execute(action="gene", gene="TP53")
        gene = result["gene"]
        assert gene["symbol"] == "TP53"
        assert gene["constraint"]["pLI"] == 0.999
        assert gene["constraint"]["oe_lof"] == 0.12
        await tool.close()


# ── HPO ────────────────────────────────────────────

class TestHPO:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self) -> None:
        tool = HPOTool()
        respx.get("https://ontology.jax.org/api/hp/search").mock(
            return_value=httpx.Response(200, json={
                "terms": [{
                    "id": "HP:0001250",
                    "name": "Seizure",
                    "definition": "A sudden episode of abnormal electrical activity in the brain.",
                    "synonyms": ["Epileptic seizure", "Convulsion"],
                }],
            })
        )
        result = await tool.execute(action="search", query="seizure")
        assert result["count"] == 1
        assert result["terms"][0]["id"] == "HP:0001250"
        assert result["terms"][0]["name"] == "Seizure"
        await tool.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_genes(self) -> None:
        tool = HPOTool()
        respx.get("https://ontology.jax.org/api/hp/terms/HP:0001250/genes").mock(
            return_value=httpx.Response(200, json={
                "genes": [
                    {"entrezGeneId": 2260, "entrezGeneSymbol": "FGFR1"},
                    {"entrezGeneId": 7157, "entrezGeneSymbol": "TP53"},
                ],
            })
        )
        result = await tool.execute(action="genes", term_id="HP:0001250")
        assert result["count"] == 2
        assert result["genes"][0]["gene_symbol"] == "FGFR1"
        await tool.close()


# ── OMIM ───────────────────────────────────────────

class TestOMIM:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMIM_API_KEY", "test_key")
        tool = OMIMTool()
        respx.get("https://api.omim.org/api/entry/search").mock(
            return_value=httpx.Response(200, json={
                "omim": {
                    "searchResponse": {
                        "totalResults": 1,
                        "entryList": [{
                            "entry": {
                                "mimNumber": 191170,
                                "titles": {"preferredTitle": "TUMOR PROTEIN p53; TP53"},
                                "status": "live",
                                "geneMap": {
                                    "geneSymbols": "TP53",
                                    "geneName": "tumor protein p53",
                                    "computedCytoLocation": "17p13.1",
                                },
                            },
                        }],
                    },
                },
            })
        )
        result = await tool.execute(action="search", query="TP53")
        assert result["count"] == 1
        assert result["entries"][0]["mim_number"] == "191170"
        assert result["entries"][0]["gene_symbols"] == "TP53"
        await tool.close()

    @pytest.mark.asyncio
    async def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OMIM_API_KEY", raising=False)
        from core.exceptions import ToolError
        tool = OMIMTool()
        with pytest.raises(ToolError, match="OMIM API key"):
            await tool.execute(action="search", query="TP53")
        await tool.close()


# ── BioGRID ────────────────────────────────────────

class TestBioGRID:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self) -> None:
        tool = BioGRIDTool()
        respx.get("https://webservice.thebiogrid.org/interactions").mock(
            return_value=httpx.Response(200, json={
                "1": {
                    "BIOGRID_INTERACTION_ID": 1,
                    "OFFICIAL_SYMBOL_A": "TP53",
                    "OFFICIAL_SYMBOL_B": "MDM2",
                    "ENTREZ_GENE_A": 7157,
                    "ENTREZ_GENE_B": 4193,
                    "EXPERIMENTAL_SYSTEM": "Two-hybrid",
                    "EXPERIMENTAL_SYSTEM_TYPE": "physical",
                    "ORGANISM_A_ID": 9606,
                    "ORGANISM_B_ID": 9606,
                    "THROUGHPUT": "Low Throughput",
                    "PUBMED_ID": 12345678,
                },
            })
        )
        result = await tool.execute(action="search", query="TP53")
        assert result["count"] == 1
        assert result["interactions"][0]["gene_a"] == "TP53"
        assert result["interactions"][0]["gene_b"] == "MDM2"
        assert result["interactions"][0]["experimental_system"] == "Two-hybrid"
        await tool.close()


# ── DepMap ─────────────────────────────────────────

class TestDepMap:
    @respx.mock
    @pytest.mark.asyncio
    async def test_gene_dependency(self) -> None:
        tool = DepMapTool()
        respx.get("https://api.depmap.org/api/v1/gene_dependency").mock(
            return_value=httpx.Response(200, json={
                "data": [
                    {"depmap_id": "ACH-000001", "cell_line_name": "A549", "lineage": "lung", "dependency": -0.85},
                    {"depmap_id": "ACH-000002", "cell_line_name": "HeLa", "lineage": "cervix", "dependency": -0.12},
                ],
            })
        )
        result = await tool.execute(action="gene_dependency", gene="TP53")
        assert result["count"] == 2
        assert result["dependencies"][0]["cell_line"] == "A549"
        assert result["dependencies"][0]["dependency_score"] == -0.85
        await tool.close()


# ── CELLxGENE ──────────────────────────────────────

class TestCellxGene:
    @respx.mock
    @pytest.mark.asyncio
    async def test_list_collections(self) -> None:
        tool = CellxGeneTool()
        respx.get("https://api.cellxgene.cziscience.com/curation/v1/collections").mock(
            return_value=httpx.Response(200, json=[
                {
                    "collection_id": "col-001",
                    "name": "Human Lung Cell Atlas",
                    "description": "Integrated atlas of the human lung.",
                    "doi": "10.1234/lung-atlas",
                },
            ])
        )
        result = await tool.execute(action="collections")
        assert result["count"] == 1
        assert result["collections"][0]["name"] == "Human Lung Cell Atlas"
        await tool.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_collection(self) -> None:
        tool = CellxGeneTool()
        respx.get("https://api.cellxgene.cziscience.com/curation/v1/collections/col-001").mock(
            return_value=httpx.Response(200, json={
                "name": "Human Lung Cell Atlas",
                "description": "Integrated atlas of the human lung.",
                "datasets": [{
                    "dataset_id": "ds-001",
                    "name": "Lung 10x scRNA-seq",
                    "organism": [{"label": "Homo sapiens"}],
                    "tissue": [{"label": "lung"}],
                    "disease": [{"label": "normal"}],
                    "cell_count": 50000,
                    "assay": [{"label": "10x 3' v3"}],
                }],
            })
        )
        result = await tool.execute(action="collection", collection_id="col-001")
        assert result["count"] == 1
        assert result["datasets"][0]["cell_count"] == 50000
        await tool.close()


# ── STRING ─────────────────────────────────────────

class TestStringDB:
    @respx.mock
    @pytest.mark.asyncio
    async def test_network(self) -> None:
        tool = StringDBTool()
        respx.get("https://string-db.org/api/json/network").mock(
            return_value=httpx.Response(200, json=[
                {
                    "preferredName_A": "TP53",
                    "preferredName_B": "MDM2",
                    "stringId_A": "9606.ENSP00000269305",
                    "stringId_B": "9606.ENSP00000258149",
                    "score": 0.999,
                    "nscore": 0, "fscore": 0, "pscore": 0,
                    "ascore": 0.9, "escore": 0.99, "dscore": 0.9, "tscore": 0.95,
                },
            ])
        )
        result = await tool.execute(action="network", query="TP53")
        assert result["count"] == 1
        assert result["interactions"][0]["protein_a"] == "TP53"
        assert result["interactions"][0]["protein_b"] == "MDM2"
        assert result["interactions"][0]["score"] == 0.999
        await tool.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_interaction_partners(self) -> None:
        tool = StringDBTool()
        respx.get("https://string-db.org/api/json/interaction_partners").mock(
            return_value=httpx.Response(200, json=[
                {"preferredName_A": "TP53", "preferredName_B": "BRCA1", "score": 0.95},
                {"preferredName_A": "TP53", "preferredName_B": "ATM", "score": 0.88},
            ])
        )
        result = await tool.execute(action="interaction_partners", query="TP53")
        assert result["count"] == 2
        assert result["interactions"][1]["protein_b"] == "ATM"
        await tool.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_enrichment(self) -> None:
        tool = StringDBTool()
        respx.get("https://string-db.org/api/json/enrichment").mock(
            return_value=httpx.Response(200, json=[
                {
                    "category": "Process",
                    "term": "GO:0006915",
                    "description": "apoptotic process",
                    "p_value": 1.2e-10,
                    "fdr": 5.5e-8,
                    "number_of_genes": 3,
                    "inputGenes": "TP53,BAX,BCL2",
                },
            ])
        )
        result = await tool.execute(action="enrichment", identifiers=["TP53", "BAX", "BCL2"])
        assert result["count"] == 1
        assert result["enrichments"][0]["term"] == "GO:0006915"
        assert result["enrichments"][0]["genes"] == ["TP53", "BAX", "BCL2"]
        await tool.close()
