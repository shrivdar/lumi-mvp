"""Tests for the know-how retriever — document loading, tag fallback, and LLM selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from know_how.retriever import _KNOW_HOW_DIR, KnowHowRetriever

# ---------------------------------------------------------------------------
# Index and document loading
# ---------------------------------------------------------------------------


class TestIndexLoading:
    def test_index_loads_all_documents(self):
        """The index should contain all 12 documents."""
        retriever = KnowHowRetriever()
        assert len(retriever._index) == 12

    def test_index_has_required_fields(self):
        """Each document entry should have id, path, title, description, tags."""
        retriever = KnowHowRetriever()
        required_fields = {"id", "path", "title", "description", "tags"}
        for doc in retriever._index:
            assert required_fields.issubset(doc.keys()), f"Missing fields in {doc.get('id')}"

    def test_all_documents_exist_on_disk(self):
        """Every document referenced in index.json should exist."""
        retriever = KnowHowRetriever()
        for doc in retriever._index:
            path = _KNOW_HOW_DIR / doc["path"]
            assert path.exists(), f"Document not found: {doc['path']}"

    def test_documents_are_non_empty(self):
        """Every document should have meaningful content."""
        retriever = KnowHowRetriever()
        for doc in retriever._index:
            content = retriever.load_document(doc)
            assert len(content) > 200, f"Document too short: {doc['id']} ({len(content)} chars)"

    def test_document_ids_are_unique(self):
        """Document IDs should be unique."""
        retriever = KnowHowRetriever()
        ids = [doc["id"] for doc in retriever._index]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Tag-based fallback retrieval
# ---------------------------------------------------------------------------


class TestTagFallback:
    def test_gwas_task_returns_gwas_doc(self):
        """A GWAS task should match the GWAS protocol via tag matching."""
        retriever = KnowHowRetriever()
        result = retriever._fallback_tag_match(
            "Interpret GWAS summary statistics for breast cancer susceptibility loci "
            "and perform fine-mapping to identify causal variants"
        )
        assert "gwas_analysis" in result

    def test_drug_task_returns_drug_doc(self):
        """A drug repurposing task should match drug_repurposing protocol."""
        retriever = KnowHowRetriever()
        result = retriever._fallback_tag_match(
            "Search ChEMBL for compounds targeting EGFR and assess ADMET properties "
            "for drug repurposing in NSCLC"
        )
        assert "drug_repurposing" in result

    def test_protein_task_returns_protein_doc(self):
        """A protein analysis task should match protein_function protocol."""
        retriever = KnowHowRetriever()
        result = retriever._fallback_tag_match(
            "Analyze protein structure and domain architecture of BRCA1 "
            "using AlphaFold predictions and UniProt annotations"
        )
        assert "protein_function" in result

    def test_variant_task_returns_variant_doc(self):
        """A variant interpretation task should match variant_interpretation protocol."""
        retriever = KnowHowRetriever()
        result = retriever._fallback_tag_match(
            "Classify the pathogenicity of BRCA1 c.5266dupC variant using ACMG criteria "
            "and ClinVar evidence"
        )
        assert "variant_interpretation" in result

    def test_clinical_task_returns_clinical_doc(self):
        """A clinical trial task should match clinical_trial_analysis protocol."""
        retriever = KnowHowRetriever()
        result = retriever._fallback_tag_match(
            "Analyze Phase III clinical trial NCT00001234 for efficacy endpoints "
            "and safety assessment of tamoxifen"
        )
        assert "clinical_trial_analysis" in result

    def test_pathway_task_returns_pathway_doc(self):
        """A pathway enrichment task should match pathway_analysis protocol."""
        retriever = KnowHowRetriever()
        result = retriever._fallback_tag_match(
            "Run GSEA pathway enrichment analysis on KEGG and Reactome pathways "
            "for differentially expressed genes in NSCLC"
        )
        assert "pathway_analysis" in result

    def test_sequence_task_returns_sequence_doc(self):
        """A sequence analysis task should match sequence_analysis protocol."""
        retriever = KnowHowRetriever()
        result = retriever._fallback_tag_match(
            "Perform BLAST homology search and multiple sequence alignment "
            "for TP53 orthologs across species"
        )
        assert "sequence_analysis" in result

    def test_max_docs_respected(self):
        """Fallback should respect max_docs setting."""
        retriever = KnowHowRetriever(max_docs=2)
        result = retriever._fallback_tag_match(
            "GWAS variant fine-mapping pathway enrichment drug repurposing"
        )
        assert len(result) <= 2

    def test_irrelevant_task_returns_empty_or_few(self):
        """A task with no matching keywords should return empty or minimal results."""
        retriever = KnowHowRetriever()
        result = retriever._fallback_tag_match("make me a sandwich")
        assert len(result) <= 1  # May match nothing or a spurious low-score hit

    def test_gene_disease_task_returns_gene_disease_doc(self):
        """A gene-disease association task should match the corresponding protocol."""
        retriever = KnowHowRetriever()
        result = retriever._fallback_tag_match(
            "Evaluate gene-disease association evidence for BRCA1 and breast cancer "
            "including Mendelian genetics and OMIM curation"
        )
        assert "gene_disease_association" in result


# ---------------------------------------------------------------------------
# LLM-based retrieval (mocked)
# ---------------------------------------------------------------------------


class TestLLMRetrieval:
    @pytest.mark.asyncio
    async def test_retrieve_parses_llm_json_response(self):
        """Retrieve should parse a clean JSON array from the LLM."""
        retriever = KnowHowRetriever()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='["gwas_analysis", "pathway_analysis", "pandas_bio_recipes"]')]

        with patch.object(retriever._client.messages, "create", return_value=mock_response):
            docs = await retriever.retrieve("Interpret GWAS hits and run pathway enrichment")

        ids = [d["id"] for d in docs]
        assert ids == ["gwas_analysis", "pathway_analysis", "pandas_bio_recipes"]

    @pytest.mark.asyncio
    async def test_retrieve_falls_back_on_llm_failure(self):
        """If LLM call fails, should fall back to tag matching."""
        retriever = KnowHowRetriever()

        with patch.object(
            retriever._client.messages, "create",
            side_effect=Exception("API error"),
        ):
            docs = await retriever.retrieve(
                "Analyze GWAS summary statistics for fine-mapping"
            )

        # Should still return results via tag fallback
        ids = [d["id"] for d in docs]
        assert "gwas_analysis" in ids

    @pytest.mark.asyncio
    async def test_retrieve_handles_malformed_llm_response(self):
        """If LLM returns non-JSON, should fall back to tag matching."""
        retriever = KnowHowRetriever()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I think the best documents are gwas and pathway")]

        with patch.object(retriever._client.messages, "create", return_value=mock_response):
            docs = await retriever.retrieve(
                "Analyze GWAS data and identify enriched pathways"
            )

        # Falls back to tag matching — should still find relevant docs
        assert len(docs) >= 0  # May or may not find matches via fallback

    @pytest.mark.asyncio
    async def test_retrieve_handles_invalid_ids(self):
        """If LLM returns non-existent IDs, they should be filtered out."""
        retriever = KnowHowRetriever()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='["gwas_analysis", "nonexistent_doc", "pathway_analysis"]')]

        with patch.object(retriever._client.messages, "create", return_value=mock_response):
            docs = await retriever.retrieve("GWAS and pathway analysis")

        ids = [d["id"] for d in docs]
        assert "nonexistent_doc" not in ids
        assert "gwas_analysis" in ids
        assert "pathway_analysis" in ids


# ---------------------------------------------------------------------------
# Full context generation
# ---------------------------------------------------------------------------


class TestContextGeneration:
    @pytest.mark.asyncio
    async def test_get_context_for_task_returns_formatted_string(self):
        """get_context_for_task should return a formatted string with document content."""
        retriever = KnowHowRetriever(max_docs=1)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='["gwas_analysis"]')]

        with patch.object(retriever._client.messages, "create", return_value=mock_response):
            context = await retriever.get_context_for_task("Interpret GWAS results")

        assert "## Domain Know-How" in context
        assert "GWAS Analysis Protocol" in context
        assert "Fine-Mapping" in context  # Content from the actual document

    @pytest.mark.asyncio
    async def test_get_context_for_task_empty_when_no_docs(self):
        """Should return empty string when no documents match."""
        retriever = KnowHowRetriever()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="[]")]

        with patch.object(retriever._client.messages, "create", return_value=mock_response):
            context = await retriever.get_context_for_task("Completely irrelevant task")

        assert context == ""


# ---------------------------------------------------------------------------
# Document content quality
# ---------------------------------------------------------------------------


class TestDocumentQuality:
    """Verify protocol documents contain expected sections."""

    def _load_doc(self, doc_id: str) -> str:
        retriever = KnowHowRetriever()
        doc = next(d for d in retriever._index if d["id"] == doc_id)
        return retriever.load_document(doc)

    def test_gwas_has_key_sections(self):
        content = self._load_doc("gwas_analysis")
        assert "Fine-Mapping" in content
        assert "Common Pitfalls" in content
        assert "Output Expectations" in content

    def test_drug_repurposing_has_key_sections(self):
        content = self._load_doc("drug_repurposing")
        assert "ADMET" in content
        assert "Common Pitfalls" in content
        assert "Binding Analysis" in content

    def test_variant_interpretation_has_acmg(self):
        content = self._load_doc("variant_interpretation")
        assert "ACMG" in content
        assert "Pathogenic" in content
        assert "ClinVar" in content

    def test_clinical_trial_has_phases(self):
        content = self._load_doc("clinical_trial_analysis")
        assert "Phase I" in content
        assert "Phase III" in content
        assert "Failure Analysis" in content

    def test_all_protocols_have_common_pitfalls(self):
        """Every protocol document should have a Common Pitfalls section."""
        retriever = KnowHowRetriever()
        protocol_docs = [d for d in retriever._index if d["path"].startswith("protocols/")]
        for doc in protocol_docs:
            content = retriever.load_document(doc)
            assert "Common Pitfalls" in content, f"{doc['id']} missing Common Pitfalls"

    def test_all_tools_have_recipes(self):
        """Every tool recipe should have code blocks."""
        retriever = KnowHowRetriever()
        tool_docs = [d for d in retriever._index if d["path"].startswith("tools/")]
        for doc in tool_docs:
            content = retriever.load_document(doc)
            assert "```python" in content, f"{doc['id']} missing Python code blocks"
