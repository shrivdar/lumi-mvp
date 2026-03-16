"""Test that tool catalog names match actual tool instance names.

This prevents the critical bug where agents receive 0 tools because
the catalog uses short names (e.g., "pubmed") while tool instances
register with suffixed names (e.g., "pubmed_search").
"""

from __future__ import annotations

import pytest

from core.models import ToolSourceType
from core.tool_registry import InMemoryToolRegistry
from integrations.registry import IntegrationsRegistry
from integrations.tool_catalog import get_catalog


class TestToolNameAlignment:
    """Verify every NATIVE tool in the catalog has a matching instance name."""

    @pytest.mark.asyncio
    async def test_catalog_native_names_match_instance_names(self) -> None:
        """All NATIVE entries in the catalog must have a corresponding
        tool instance with the same name in the IntegrationsRegistry."""
        # Bootstrap all native tool instances
        tool_reg = InMemoryToolRegistry()
        reg = IntegrationsRegistry(redis=None, tool_registry=tool_reg)
        await reg.bootstrap()

        instance_names = {t.name for t in reg.list_instances()}

        # Get all NATIVE entries from the catalog
        catalog = get_catalog()
        native_catalog_names = {
            entry.name
            for entry in catalog
            if entry.source_type == ToolSourceType.NATIVE
        }

        # Every native catalog name must exist as an instance name
        missing = native_catalog_names - instance_names
        assert not missing, (
            f"Catalog NATIVE tool names not found in registry instances: {missing}. "
            f"Catalog names must match the `name` attribute on each tool class."
        )

    @pytest.mark.asyncio
    async def test_instance_names_in_catalog(self) -> None:
        """Every bootstrapped tool instance should have a catalog entry."""
        tool_reg = InMemoryToolRegistry()
        reg = IntegrationsRegistry(redis=None, tool_registry=tool_reg)
        await reg.bootstrap()

        instance_names = {t.name for t in reg.list_instances()}

        catalog = get_catalog()
        catalog_names = {entry.name for entry in catalog}

        # python_repl is a special tool always injected by research_loop,
        # not managed through the catalog
        SPECIAL_TOOLS = {"python_repl"}
        missing = instance_names - catalog_names - SPECIAL_TOOLS
        assert not missing, (
            f"Tool instances not found in catalog: {missing}. "
            f"Every tool instance must have a corresponding catalog entry."
        )

    def test_no_duplicate_catalog_names(self) -> None:
        """Catalog should not have duplicate tool names."""
        catalog = get_catalog()
        names = [entry.name for entry in catalog]
        duplicates = {n for n in names if names.count(n) > 1}
        assert not duplicates, f"Duplicate tool names in catalog: {duplicates}"
