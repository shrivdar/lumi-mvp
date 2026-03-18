"""DynamicToolRegistry — manages runtime collection of dynamically created tools.

Provides in-memory registry with optional Postgres persistence via DynamicToolRow.
Tool names are prefixed with ``dyn_`` to distinguish from native tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from core.models import DynamicToolSpec, DynamicToolStatus, ToolRegistryEntry, ToolSourceType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from integrations.dynamic.dynamic_tool import DynamicTool

logger = structlog.get_logger(__name__)

# Prefix for all dynamic tool names in the registry
DYN_PREFIX = "dyn_"


def _ensure_prefix(name: str) -> str:
    """Ensure a tool name has the ``dyn_`` prefix."""
    if name.startswith(DYN_PREFIX):
        return name
    return f"{DYN_PREFIX}{name}"


class DynamicToolRegistry:
    """Runtime collection of dynamically created tools.

    Tracks ``DynamicTool`` instances in memory and optionally persists them
    to the ``dynamic_tools`` Postgres table.  Tools that exceed quality
    thresholds (``success_rate > 0.7`` and ``usage_count > 2``) are
    automatically loaded from the database on startup.
    """

    def __init__(self) -> None:
        self._tools: dict[str, DynamicTool] = {}
        self._specs: dict[str, DynamicToolSpec] = {}
        self._usage: dict[str, dict[str, int]] = {}  # name -> {usage_count, success_count}

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def register(self, tool: DynamicTool) -> None:
        """Add a dynamic tool to the registry.

        The tool's name is normalized with a ``dyn_`` prefix.
        """
        name = _ensure_prefix(tool.name)
        self._tools[name] = tool
        self._specs[name] = tool.spec
        self._usage.setdefault(name, {"usage_count": 0, "success_count": 0})
        logger.info("dynamic_tool_registered", tool=name, category=tool.category)

    def get_tool(self, name: str) -> DynamicTool | None:
        """Lookup a dynamic tool by name (with or without prefix)."""
        return self._tools.get(_ensure_prefix(name))

    def get_tools(self) -> list[DynamicTool]:
        """Return all registered dynamic tools."""
        return list(self._tools.values())

    def get_specs(self) -> list[DynamicToolSpec]:
        """Return all registered tool specs."""
        return list(self._specs.values())

    def get_registry_entries(self) -> list[ToolRegistryEntry]:
        """Return ToolRegistryEntry objects for all dynamic tools.

        Used to inject dynamic tools into the SwarmComposer's tool catalog.
        """
        entries: list[ToolRegistryEntry] = []
        for name, tool in self._tools.items():
            entries.append(ToolRegistryEntry(
                name=name,
                description=tool.description,
                source_type=ToolSourceType.DYNAMIC,
                category=tool.category,
                capabilities=tool.spec.capabilities,
            ))
        return entries

    def record_usage(self, name: str, *, success: bool) -> None:
        """Record a tool usage outcome for quality tracking."""
        key = _ensure_prefix(name)
        if key not in self._usage:
            self._usage[key] = {"usage_count": 0, "success_count": 0}
        self._usage[key]["usage_count"] += 1
        if success:
            self._usage[key]["success_count"] += 1

    def success_rate(self, name: str) -> float:
        """Return the success rate for a tool."""
        key = _ensure_prefix(name)
        stats = self._usage.get(key, {"usage_count": 0, "success_count": 0})
        if stats["usage_count"] == 0:
            return 1.0
        return stats["success_count"] / stats["usage_count"]

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    # ------------------------------------------------------------------
    # Postgres persistence
    # ------------------------------------------------------------------

    async def persist(self, tool: DynamicTool, db: AsyncSession) -> None:
        """Save or update a dynamic tool in the ``dynamic_tools`` table."""
        from db.tables import DynamicToolRow

        name = _ensure_prefix(tool.name)
        stats = self._usage.get(name, {"usage_count": 0, "success_count": 0})
        usage_count = stats["usage_count"]
        success_count = stats["success_count"]
        rate = success_count / usage_count if usage_count > 0 else 1.0

        # Upsert: check if exists first
        from sqlalchemy import select
        stmt = select(DynamicToolRow).where(DynamicToolRow.name == name)
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.wrapper_code = tool.spec.wrapper_code
            existing.description = tool.spec.description
            existing.api_base_url = tool.spec.api_base_url
            existing.test_code = tool.spec.test_code
            existing.category = tool.spec.category
            existing.capabilities = tool.spec.capabilities
            existing.example_tasks = tool.spec.example_tasks
            existing.parameters = tool.spec.parameters
            existing.status = tool.spec.status.value
            existing.usage_count = usage_count
            existing.success_count = success_count
            existing.success_rate = rate
        else:
            row = DynamicToolRow(
                name=name,
                description=tool.spec.description,
                api_base_url=tool.spec.api_base_url,
                wrapper_code=tool.spec.wrapper_code,
                test_code=tool.spec.test_code,
                category=tool.spec.category,
                capabilities=tool.spec.capabilities,
                example_tasks=tool.spec.example_tasks,
                parameters=tool.spec.parameters,
                status=tool.spec.status.value,
                created_by=tool.spec.created_by,
                usage_count=usage_count,
                success_count=success_count,
                success_rate=rate,
            )
            db.add(row)

        await db.flush()
        logger.info("dynamic_tool_persisted", tool=name)

    async def load_persisted(self, db: AsyncSession) -> int:
        """Load high-quality tools from the ``dynamic_tools`` table.

        Only loads tools where ``success_rate > 0.7`` and ``usage_count > 2``.

        Returns:
            Number of tools loaded.
        """
        from sqlalchemy import select

        from db.tables import DynamicToolRow
        from integrations.dynamic.dynamic_tool import DynamicTool

        stmt = (
            select(DynamicToolRow)
            .where(DynamicToolRow.success_rate > 0.7)
            .where(DynamicToolRow.usage_count > 2)
            .where(DynamicToolRow.status == "REGISTERED")
        )
        result = await db.execute(stmt)
        rows = result.scalars().all()

        count = 0
        for row in rows:
            if row.name in self._tools:
                continue  # already loaded

            spec = DynamicToolSpec(
                name=row.name.removeprefix(DYN_PREFIX),
                description=row.description,
                api_base_url=row.api_base_url,
                wrapper_code=row.wrapper_code,
                test_code=row.test_code,
                category=row.category,
                capabilities=row.capabilities or [],
                example_tasks=row.example_tasks or [],
                parameters=row.parameters or {},
                status=DynamicToolStatus.REGISTERED,
                created_by=row.created_by,
            )

            tool = DynamicTool(spec=spec, repl_tool=None)
            self._tools[row.name] = tool
            self._specs[row.name] = spec
            self._usage[row.name] = {
                "usage_count": row.usage_count,
                "success_count": row.success_count,
            }
            count += 1

        logger.info("dynamic_tools_loaded", count=count)
        return count
