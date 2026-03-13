"""Tests for the Yami/ESM protein modelling interface."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from core.exceptions import ToolError
from world_model.yami import YamiClient


@pytest.fixture()
def yami() -> YamiClient:
    """Yami client with HF backend and no real API token."""
    return YamiClient(backend="huggingface", hf_token="test-token")


# ═══════════════════════════════════════════════════════════════════════════
# Sequence validation
# ═══════════════════════════════════════════════════════════════════════════


class TestSequenceValidation:
    def test_valid_sequence(self) -> None:
        assert YamiClient.validate_sequence("ACDEFGHIKLMNPQRSTVWY") is True

    def test_invalid_characters(self) -> None:
        assert YamiClient.validate_sequence("ACDE123") is False

    def test_empty_sequence(self) -> None:
        assert YamiClient.validate_sequence("") is False

    def test_lowercase_accepted(self) -> None:
        assert YamiClient.validate_sequence("acdef") is True


# ═══════════════════════════════════════════════════════════════════════════
# PDB parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestPDBParsing:
    def test_extract_plddt_from_pdb(self) -> None:
        pdb_text = (
            "ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00 72.30           N\n"
            "ATOM      2  CA  ALA A   1      26.266  25.413   2.842  1.00 72.30           C\n"
            "ATOM     10  N   GLY A   2      24.106  24.610   3.474  1.00 85.60           N\n"
            "ATOM     11  CA  GLY A   2      22.870  25.345   3.642  1.00 85.60           C\n"
            "ATOM     20  N   VAL A   3      21.840  23.410   4.874  1.00 91.20           N\n"
            "ATOM     21  CA  VAL A   3      20.660  24.240   5.042  1.00 91.20           C\n"
        )
        scores = YamiClient._extract_plddt_from_pdb(pdb_text)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(72.30)
        assert scores[1] == pytest.approx(85.60)
        assert scores[2] == pytest.approx(91.20)

    def test_empty_pdb(self) -> None:
        assert YamiClient._extract_plddt_from_pdb("") == []

    def test_pdb_no_atoms(self) -> None:
        pdb_text = "HEADER  test\nTITLE  test protein\nEND\n"
        assert YamiClient._extract_plddt_from_pdb(pdb_text) == []


# ═══════════════════════════════════════════════════════════════════════════
# Caching
# ═══════════════════════════════════════════════════════════════════════════


class TestCaching:
    @pytest.mark.asyncio
    async def test_logits_cached(self, yami: YamiClient) -> None:
        fake_logits = np.random.rand(10, 33).astype(np.float32)
        cache_key = yami._cache_key("logits", "ACDEFGHIKL")
        yami._cache[cache_key] = fake_logits

        result = await yami.get_logits("ACDEFGHIKL")
        np.testing.assert_array_equal(result, fake_logits)

    @pytest.mark.asyncio
    async def test_embeddings_cached(self, yami: YamiClient) -> None:
        fake_emb = np.random.rand(1280).astype(np.float32)
        cache_key = yami._cache_key("embeddings", "ACDEF")
        yami._cache[cache_key] = fake_emb

        result = await yami.get_embeddings("ACDEF")
        np.testing.assert_array_equal(result, fake_emb)

    @pytest.mark.asyncio
    async def test_structure_cached(self, yami: YamiClient) -> None:
        fake_result = {
            "pdb": "ATOM...",
            "plddt_mean": 85.0,
            "plddt_per_residue": [85.0],
            "sequence_length": 5,
        }
        cache_key = yami._cache_key("structure", "ACDEF")
        yami._cache[cache_key] = fake_result

        result = await yami.predict_structure("ACDEF")
        assert result["plddt_mean"] == 85.0


# ═══════════════════════════════════════════════════════════════════════════
# HuggingFace API backend (mocked)
# ═══════════════════════════════════════════════════════════════════════════


class TestHFBackend:
    @pytest.mark.asyncio
    async def test_hf_logits_success(self, yami: YamiClient) -> None:
        fake_response = [[list(range(33)) for _ in range(5)]]  # (1, 5, 33)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        yami._http = mock_client

        result = await yami.get_logits("ACDEF")
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5

    @pytest.mark.asyncio
    async def test_hf_embeddings_success(self, yami: YamiClient) -> None:
        # Shape: (1, 3, 1280) → mean → (1280,)
        fake_response = [[[float(i)] * 4 for i in range(3)]]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        yami._http = mock_client

        result = await yami.get_embeddings("ACD")
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    @pytest.mark.asyncio
    async def test_hf_structure_success(self, yami: YamiClient) -> None:
        pdb_text = (
            "ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00 72.30           N\n"
            "ATOM      2  CA  ALA A   1      26.266  25.413   2.842  1.00 72.30           C\n"
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = pdb_text

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        yami._http = mock_client

        result = await yami.predict_structure("A")
        assert "pdb" in result
        assert result["sequence_length"] == 1

    @pytest.mark.asyncio
    async def test_hf_api_error_raises_tool_error(self, yami: YamiClient) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Model is loading"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        yami._http = mock_client

        with pytest.raises(ToolError, match="HuggingFace API error"):
            await yami.get_logits("ACDEF")


# ═══════════════════════════════════════════════════════════════════════════
# Derived methods
# ═══════════════════════════════════════════════════════════════════════════


class TestDerivedMethods:
    @pytest.mark.asyncio
    async def test_compute_fitness(self, yami: YamiClient) -> None:
        # Mock logits via cache
        wt_logits = np.random.rand(5, 33).astype(np.float32)
        mut_logits = np.random.rand(5, 33).astype(np.float32)

        yami._cache[yami._cache_key("logits", "ACDEF")] = wt_logits
        yami._cache[yami._cache_key("logits", "ACGEF")] = mut_logits

        result = await yami.compute_fitness("ACDEF", "ACGEF")
        assert "fitness_delta" in result
        assert "mutations" in result
        assert len(result["mutations"]) == 1
        assert result["mutations"][0]["position"] == 2
        assert result["mutations"][0]["wild_type_aa"] == "D"
        assert result["mutations"][0]["mutant_aa"] == "G"

    @pytest.mark.asyncio
    async def test_compute_similarity(self, yami: YamiClient) -> None:
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        yami._cache[yami._cache_key("embeddings", "AAA")] = emb_a
        yami._cache[yami._cache_key("embeddings", "BBB")] = emb_b

        sim = await yami.compute_similarity("AAA", "BBB")
        assert sim == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_compute_similarity_orthogonal(self, yami: YamiClient) -> None:
        emb_a = np.array([1.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.0, 1.0], dtype=np.float32)

        yami._cache[yami._cache_key("embeddings", "XX")] = emb_a
        yami._cache[yami._cache_key("embeddings", "YY")] = emb_b

        sim = await yami.compute_similarity("XX", "YY")
        assert sim == pytest.approx(0.0, abs=1e-6)
