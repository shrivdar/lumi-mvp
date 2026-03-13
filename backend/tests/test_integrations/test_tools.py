"""Tests for remaining API tools — UniProt, KEGG, Reactome, MyGene, ChEMBL, ClinicalTrials, ESM."""

from __future__ import annotations

import httpx
import pytest
import respx

from integrations.chembl import ChEMBLTool
from integrations.clinicaltrials import ClinicalTrialsTool
from integrations.esm import ESMTool
from integrations.kegg import KEGGTool
from integrations.mygene import MyGeneTool
from integrations.reactome import ReactomeTool
from integrations.uniprot import UniProtTool

# ── UniProt ─────────────────────────────────────────

class TestUniProt:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self) -> None:
        tool = UniProtTool()
        respx.get("https://rest.uniprot.org/uniprotkb/search").mock(
            return_value=httpx.Response(200, json={
                "results": [{
                    "primaryAccession": "Q5T4S7",
                    "uniProtkbId": "UBB_HUMAN",
                    "proteinDescription": {"recommendedName": {"fullName": {"value": "Ubiquitin B"}}},
                    "genes": [{"geneName": {"value": "UBB"}}],
                    "organism": {"scientificName": "Homo sapiens"},
                    "sequence": {"value": "MQIFVK", "length": 6, "molWeight": 700},
                    "uniProtKBCrossReferences": [],
                    "comments": [],
                }]
            })
        )
        result = await tool.execute(action="search", query="ubiquitin human")
        assert len(result["entries"]) == 1
        assert result["entries"][0]["accession"] == "Q5T4S7"
        assert result["entries"][0]["protein_name"] == "Ubiquitin B"
        await tool.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_entry(self) -> None:
        tool = UniProtTool()
        respx.get("https://rest.uniprot.org/uniprotkb/P12345.json").mock(
            return_value=httpx.Response(200, json={
                "primaryAccession": "P12345",
                "proteinDescription": {"recommendedName": {"fullName": {"value": "Aspartate aminotransferase"}}},
                "genes": [],
                "organism": {"scientificName": "Oryctolagus cuniculus"},
                "sequence": {"value": "MKL", "length": 3},
                "uniProtKBCrossReferences": [
                    {"database": "PDB", "id": "1AAT", "properties": []},
                    {"database": "GO", "id": "GO:0006520", "properties": [
                        {"key": "GoTerm", "value": "amino acid metabolic process"},
                        {"key": "GoEvidenceType", "value": "IEA"},
                    ]},
                ],
                "comments": [{"commentType": "FUNCTION", "texts": [{"value": "Catalyzes transamination."}]}],
            })
        )
        result = await tool.execute(action="entry", accession="P12345")
        entry = result["entry"]
        assert entry["accession"] == "P12345"
        assert entry["pdb_ids"] == ["1AAT"]
        assert entry["function"] == "Catalyzes transamination."
        assert any(g["id"] == "GO:0006520" for g in entry["go_terms"])
        await tool.close()


# ── KEGG ────────────────────────────────────────────

class TestKEGG:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self) -> None:
        tool = KEGGTool()
        respx.get("https://rest.kegg.jp/find/pathway/apoptosis").mock(
            return_value=httpx.Response(200, text="path:hsa04210\tApoptosis - Homo sapiens\n")
        )
        result = await tool.execute(action="search", database="pathway", query="apoptosis")
        assert result["count"] == 1
        assert result["results"][0]["id"] == "path:hsa04210"
        await tool.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_entry(self) -> None:
        tool = KEGGTool()
        respx.get("https://rest.kegg.jp/get/hsa:7157").mock(
            return_value=httpx.Response(
                200, text="ENTRY       7157\nNAME        TP53\nDEFINITION  tumor protein p53\n///\n",
            )
        )
        result = await tool.execute(action="get", entry_id="hsa:7157")
        assert result["data"]["ENTRY"] == "7157"
        assert result["data"]["NAME"] == "TP53"
        await tool.close()


# ── Reactome ────────────────────────────────────────

class TestReactome:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self) -> None:
        tool = ReactomeTool()
        respx.get("https://reactome.org/ContentService/search/query").mock(
            return_value=httpx.Response(200, json={
                "results": [{
                    "entries": [{
                        "stId": "R-HSA-109581",
                        "name": "Apoptosis",
                        "species": ["Homo sapiens"],
                        "summation": "Programmed cell death",
                        "exactType": "Pathway",
                    }]
                }]
            })
        )
        result = await tool.execute(action="search", query="apoptosis")
        assert result["count"] == 1
        assert result["pathways"][0]["stable_id"] == "R-HSA-109581"
        await tool.close()


# ── MyGene ──────────────────────────────────────────

class TestMyGene:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self) -> None:
        tool = MyGeneTool()
        respx.get("https://mygene.info/v3/query").mock(
            return_value=httpx.Response(200, json={
                "total": 1,
                "hits": [{
                    "_id": "7157",
                    "symbol": "TP53",
                    "name": "tumor protein p53",
                    "entrezgene": 7157,
                    "ensembl": {"gene": "ENSG00000141510"},
                    "uniprot": {"Swiss-Prot": "P04637"},
                    "type_of_gene": "protein-coding",
                    "summary": "Tumor suppressor",
                    "genomic_pos": {"chr": "17", "start": 7668402, "end": 7687550, "strand": -1},
                    "go": {"BP": [{"id": "GO:0006915", "term": "apoptotic process"}]},
                    "pathway": {"kegg": [{"id": "hsa04115", "name": "p53 signaling pathway"}]},
                }]
            })
        )
        result = await tool.execute(action="search", query="TP53")
        gene = result["genes"][0]
        assert gene["symbol"] == "TP53"
        assert gene["entrez_id"] == "7157"
        assert gene["uniprot_id"] == "P04637"
        assert gene["kegg_pathways"][0]["name"] == "p53 signaling pathway"
        await tool.close()


# ── ChEMBL ──────────────────────────────────────────

class TestChEMBL:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search_compound(self) -> None:
        tool = ChEMBLTool()
        respx.get("https://www.ebi.ac.uk/chembl/api/data/molecule/search.json").mock(
            return_value=httpx.Response(200, json={
                "molecules": [{
                    "molecule_chembl_id": "CHEMBL25",
                    "pref_name": "ASPIRIN",
                    "molecule_type": "Small molecule",
                    "max_phase": 4,
                    "molecule_properties": {"full_mwt": "180.16", "alogp": "1.31"},
                    "molecule_structures": {"canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
                }]
            })
        )
        result = await tool.execute(action="search_compound", query="aspirin")
        assert result["count"] == 1
        assert result["compounds"][0]["chembl_id"] == "CHEMBL25"
        assert result["compounds"][0]["name"] == "ASPIRIN"
        await tool.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_mechanisms(self) -> None:
        tool = ChEMBLTool()
        respx.get("https://www.ebi.ac.uk/chembl/api/data/mechanism.json").mock(
            return_value=httpx.Response(200, json={
                "mechanisms": [{
                    "mechanism_of_action": "Cyclooxygenase inhibitor",
                    "action_type": "INHIBITOR",
                    "target_chembl_id": "CHEMBL2094253",
                    "target_name": "Cyclooxygenase",
                }]
            })
        )
        result = await tool.execute(action="mechanisms", chembl_id="CHEMBL25")
        assert result["count"] == 1
        assert result["mechanisms"][0]["mechanism_of_action"] == "Cyclooxygenase inhibitor"
        await tool.close()


# ── ClinicalTrials ──────────────────────────────────

class TestClinicalTrials:
    @respx.mock
    @pytest.mark.asyncio
    async def test_search(self) -> None:
        tool = ClinicalTrialsTool()
        respx.get("https://clinicaltrials.gov/api/v2/studies").mock(
            return_value=httpx.Response(200, json={
                "totalCount": 1,
                "studies": [{
                    "protocolSection": {
                        "identificationModule": {"nctId": "NCT06000001", "briefTitle": "B7-H3 Trial"},
                        "statusModule": {"overallStatus": "RECRUITING"},
                        "designModule": {"phases": ["PHASE3"], "studyType": "INTERVENTIONAL"},
                        "armsInterventionsModule": {
                            "interventions": [{"name": "Anti-B7-H3", "type": "DRUG", "description": "monoclonal Ab"}]
                        },
                        "conditionsModule": {"conditions": ["NSCLC"]},
                        "descriptionModule": {"briefSummary": "Phase 3 trial of anti-B7-H3"},
                        "outcomesModule": {"primaryOutcomes": [{"measure": "OS", "timeFrame": "24 months"}]},
                        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "TestPharma"}},
                        "eligibilityModule": {},
                    }
                }]
            })
        )
        result = await tool.execute(action="search", query="B7-H3 NSCLC")
        assert result["total_count"] == 1
        trial = result["trials"][0]
        assert trial["nct_id"] == "NCT06000001"
        assert trial["status"] == "RECRUITING"
        assert trial["conditions"] == ["NSCLC"]
        assert trial["interventions"][0]["name"] == "Anti-B7-H3"
        await tool.close()


# ── ESM ─────────────────────────────────────────────

class TestESM:
    @respx.mock
    @pytest.mark.asyncio
    async def test_fill_mask(self) -> None:
        tool = ESMTool()
        respx.post(url__regex=r".*huggingface.*").mock(
            return_value=httpx.Response(200, json=[
                {"token_str": "A", "score": 0.8, "sequence": "MALWMRLLPLLALLAL"},
                {"token_str": "G", "score": 0.1, "sequence": "MGLWMRLLPLLALLAL"},
            ])
        )
        result = await tool.execute(action="fill_mask", sequence="M<mask>LWMRLLPLLALLAL")
        assert len(result["predictions"]) == 2
        assert result["predictions"][0]["token"] == "A"
        assert result["predictions"][0]["score"] == 0.8
        await tool.close()

    def test_validate_sequence_rejects_invalid(self) -> None:
        from core.exceptions import ToolError
        with pytest.raises(ToolError, match="Invalid amino acid"):
            ESMTool._validate_sequence("ACDEFX123")

    def test_validate_sequence_accepts_valid(self) -> None:
        ESMTool._validate_sequence("ACDEFGHIKLMNPQRSTVWY")  # no error
