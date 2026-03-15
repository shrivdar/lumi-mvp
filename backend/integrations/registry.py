"""Integrations registry — thin wrapper that bootstraps native tools into the core ToolRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from core.tool_registry import InMemoryToolRegistry

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from integrations.base_tool import BaseTool

logger = structlog.get_logger(__name__)


class IntegrationsRegistry:
    """Manages native tool instances and registers them with the core ToolRegistry.

    Usage::

        registry = IntegrationsRegistry(redis=redis_client)
        await registry.bootstrap()          # instantiate + register all native tools
        tool = registry.get_instance("pubmed_search")
        result = await tool.execute(query="B7-H3 NSCLC")
    """

    def __init__(
        self,
        redis: aioredis.Redis | None = None,
        tool_registry: InMemoryToolRegistry | None = None,
    ) -> None:
        self._redis = redis
        self._tool_registry = tool_registry or InMemoryToolRegistry()
        self._instances: dict[str, BaseTool] = {}

    @property
    def tool_registry(self) -> InMemoryToolRegistry:
        return self._tool_registry

    async def bootstrap(self, **overrides: Any) -> None:
        """Instantiate all native tools and register them."""
        from integrations.biogrid import BioGRIDTool
        from integrations.cellxgene import CellxGeneTool
        from integrations.chembl import ChEMBLTool
        from integrations.clinicaltrials import ClinicalTrialsTool
        from integrations.clinvar import ClinVarTool
        from integrations.depmap import DepMapTool
        from integrations.esm import ESMTool
        from integrations.gnomad import GnomADTool
        from integrations.gtex import GTExTool
        from integrations.hpo import HPOTool
        from integrations.kegg import KEGGTool
        from integrations.mygene import MyGeneTool
        from integrations.omim import OMIMTool
        from integrations.opentargets import OpenTargetsTool
        from integrations.pubmed import PubMedTool
        from integrations.python_repl import PythonREPLTool
        from integrations.reactome import ReactomeTool
        from integrations.semantic_scholar import SemanticScholarTool
        from integrations.slack import SlackTool
        from integrations.string_db import StringDBTool
        from integrations.uniprot import UniProtTool

        tool_classes: list[type[BaseTool]] = [
            PubMedTool,
            SemanticScholarTool,
            UniProtTool,
            KEGGTool,
            ReactomeTool,
            MyGeneTool,
            ChEMBLTool,
            ClinicalTrialsTool,
            ESMTool,
            SlackTool,
            PythonREPLTool,
            # --- New tools (tool expansion) ---
            OpenTargetsTool,
            ClinVarTool,
            GTExTool,
            GnomADTool,
            HPOTool,
            OMIMTool,
            BioGRIDTool,
            DepMapTool,
            CellxGeneTool,
            StringDBTool,
        ]

        for cls in tool_classes:
            instance = cls(redis=self._redis, registry=self._tool_registry, **overrides)
            self._instances[instance.name] = instance

        logger.info("integrations_bootstrapped", tool_count=len(self._instances))

    def get_instance(self, name: str) -> BaseTool | None:
        return self._instances.get(name)

    def list_instances(self) -> list[BaseTool]:
        return list(self._instances.values())

    async def close_all(self) -> None:
        for tool in self._instances.values():
            await tool.close()
