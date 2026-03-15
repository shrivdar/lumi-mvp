"""External API integrations with caching, rate limiting, and retry.

Provides:
- BaseTool framework (cache, rate-limit, retry, audit)
- 10 native API clients (PubMed, Semantic Scholar, UniProt, KEGG, Reactome, MyGene, ChEMBL, ClinicalTrials, ESM, Slack)
- MCP protocol client for external tool servers
- Container tool manager for sandboxed tools
- IntegrationsRegistry for bootstrap and unified access
"""

from integrations.base_tool import BaseTool
from integrations.python_repl import PythonREPLTool
from integrations.registry import IntegrationsRegistry

__all__ = ["BaseTool", "IntegrationsRegistry", "PythonREPLTool"]
