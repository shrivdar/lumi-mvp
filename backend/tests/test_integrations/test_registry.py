"""Tests for IntegrationsRegistry — bootstrap and instance management."""

from __future__ import annotations

import pytest

from core.tool_registry import InMemoryToolRegistry
from integrations.registry import IntegrationsRegistry


class TestIntegrationsRegistry:
    @pytest.mark.asyncio
    async def test_bootstrap_registers_all_tools(self) -> None:
        tool_reg = InMemoryToolRegistry()
        reg = IntegrationsRegistry(redis=None, tool_registry=tool_reg)
        await reg.bootstrap()

        instances = reg.list_instances()
        assert len(instances) == 10

        # All tools should be in the core registry
        assert tool_reg.tool_count == 10

        # Spot-check names
        names = {t.name for t in instances}
        assert "pubmed_search" in names
        assert "semantic_scholar_search" in names
        assert "uniprot_search" in names
        assert "kegg_search" in names
        assert "reactome_search" in names
        assert "mygene_search" in names
        assert "chembl_search" in names
        assert "clinicaltrials_search" in names
        assert "esm_predict" in names
        assert "slack_hitl" in names

    @pytest.mark.asyncio
    async def test_get_instance(self) -> None:
        reg = IntegrationsRegistry()
        await reg.bootstrap()
        pubmed = reg.get_instance("pubmed_search")
        assert pubmed is not None
        assert pubmed.name == "pubmed_search"

    @pytest.mark.asyncio
    async def test_get_nonexistent_instance(self) -> None:
        reg = IntegrationsRegistry()
        await reg.bootstrap()
        assert reg.get_instance("nonexistent") is None

    @pytest.mark.asyncio
    async def test_close_all(self) -> None:
        reg = IntegrationsRegistry()
        await reg.bootstrap()
        await reg.close_all()  # should not raise
