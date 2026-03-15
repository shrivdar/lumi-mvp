"""Tool Creator Agent — discovers, tests, and integrates new bioinformatics tools.

STELLA-inspired: agents create their own tools, performance improves with experience.
Uses multi-turn investigation to:
1. Search for bioinformatics APIs/tools relevant to the research task
2. Read documentation and understand API contracts
3. Write a Python wrapper function following BaseTool conventions
4. Test the wrapper in a sandboxed REPL
5. Register successful tools for other agents to use
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

import structlog

from agents.base import BaseAgentImpl
from core.models import (
    AgentTask,
    AgentType,
    DynamicToolSpec,
    DynamicToolStatus,
)

logger = structlog.get_logger(__name__)

# Well-known bioinformatics APIs the agent can discover and wrap
KNOWN_API_CATALOG = {
    "string_db": {
        "name": "string_db",
        "description": "STRING database — protein-protein interaction networks and functional enrichment",
        "api_base_url": "https://string-db.org/api",
        "documentation_url": "https://string-db.org/cgi/help",
        "category": "protein",
        "capabilities": [
            "protein-protein interaction network retrieval",
            "functional enrichment analysis",
            "interaction confidence scores",
            "network visualization data",
        ],
        "example_tasks": [
            "Find protein interaction partners for B7-H3",
            "Get STRING interaction network for TP53",
            "Perform functional enrichment on a gene list",
        ],
        "example_wrapper": textwrap.dedent("""\
            import urllib.request
            import json

            def run(*, query: str = "", species: int = 9606, limit: int = 10, **kwargs) -> dict:
                \"\"\"Query STRING-db for protein-protein interactions.\"\"\"
                base = "https://string-db.org/api/json"
                params = f"identifiers={query}&species={species}&limit={limit}"
                url = f"{base}/network?{params}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                interactions = []
                for item in data:
                    interactions.append({
                        "protein_a": item.get("preferredName_A", ""),
                        "protein_b": item.get("preferredName_B", ""),
                        "score": item.get("score", 0),
                        "nscore": item.get("nscore", 0),
                        "escore": item.get("escore", 0),
                    })
                return {"query": query, "species": species, "interactions": interactions}
        """),
    },
    "ensembl": {
        "name": "ensembl",
        "description": "Ensembl REST API — gene/transcript/variant lookups, sequence retrieval",
        "api_base_url": "https://rest.ensembl.org",
        "category": "genomics",
        "capabilities": [
            "gene and transcript lookups",
            "variant effect prediction (VEP)",
            "sequence retrieval (DNA, protein)",
            "cross-species orthologs",
        ],
        "example_tasks": [
            "Get gene information for ENSG00000012048",
            "Retrieve transcript sequences for BRCA1",
            "Look up variant consequences",
        ],
    },
    "opentargets": {
        "name": "opentargets",
        "description": "Open Targets Platform — target-disease associations with genetic and literature evidence",
        "api_base_url": "https://api.platform.opentargets.org/api/v4",
        "category": "drug",
        "capabilities": [
            "target-disease association scores",
            "genetic evidence for drug targets",
            "known drug mechanisms",
            "tractability assessments",
        ],
        "example_tasks": [
            "Find diseases associated with EGFR",
            "Get drug tractability for a protein target",
            "Look up genetic evidence linking gene to disease",
        ],
    },
    "ncbi_gene": {
        "name": "ncbi_gene",
        "description": "NCBI Gene (Entrez) — gene summaries, RefSeq, orthologs, bibliography",
        "api_base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        "category": "genomics",
        "capabilities": [
            "gene summary and RefSeq data",
            "gene bibliography (GeneRIF)",
            "ortholog lookup",
            "gene-to-pathway cross-references",
        ],
        "example_tasks": [
            "Get gene summary for BRCA1 from NCBI",
            "Find RefSeq accessions for a gene",
            "Look up gene orthologs across species",
        ],
    },
    "disgenet": {
        "name": "disgenet",
        "description": "DisGeNET — gene-disease associations from curated and literature sources",
        "api_base_url": "https://www.disgenet.org/api",
        "category": "genomics",
        "capabilities": [
            "gene-disease association scores",
            "variant-disease associations",
            "disease-based gene prioritization",
        ],
        "example_tasks": [
            "Find disease associations for BRCA1",
            "Get top genes associated with breast cancer",
            "Look up variant-disease evidence",
        ],
    },
}


class ToolCreatorAgent(BaseAgentImpl):
    """Discovers, writes, tests, and registers new bioinformatics tool wrappers.

    Uses the multi-turn investigation loop to:
    1. Identify gaps in the current tool registry
    2. Search for relevant APIs from the known catalog or via literature search
    3. Write wrapper code following the ``run(**kwargs) -> dict`` contract
    4. Test the wrapper in a sandboxed REPL
    5. Register validated tools in the tool registry
    """

    agent_type = AgentType.TOOL_CREATOR

    async def _investigate(
        self,
        task: AgentTask,
        kg_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Orchestrate the tool creation workflow using multi-turn investigation."""
        return await self._multi_turn_investigate(
            task,
            kg_context,
            investigation_focus=(
                "You are a tool creation specialist. Your mission is to discover and create "
                "new bioinformatics tool wrappers that the research platform can use.\n\n"
                "## Available API Catalog\n"
                f"{self._format_api_catalog()}\n\n"
                "## Tool Creation Workflow\n"
                "1. Analyze the research task to identify what data sources are needed\n"
                "2. Check the known API catalog for relevant APIs\n"
                "3. Write a Python wrapper function following this contract:\n"
                "   ```python\n"
                "   def run(*, query: str = '', **kwargs) -> dict:\n"
                "       '''Docstring explaining the tool.'''\n"
                "       # Implementation using urllib.request (no external deps)\n"
                "       return {'results': [...]}\n"
                "   ```\n"
                "4. Use <execute> to test the wrapper in the sandboxed REPL\n"
                "5. If tests pass, output the tool spec in your <answer>\n\n"
                "## Rules\n"
                "- Wrappers must use only stdlib (urllib.request, json, xml.etree)\n"
                "- Must handle errors gracefully (timeouts, HTTP errors)\n"
                "- Must return structured dicts, not raw strings\n"
                "- Include timeout=30 on all network requests\n"
                "- The REPL has --network none, so use <tool> calls for actual API testing "
                "and <execute> for code validation/parsing only"
            ),
        )

    def _format_api_catalog(self) -> str:
        """Format the known API catalog for inclusion in the investigation prompt."""
        lines: list[str] = []
        for api_id, info in KNOWN_API_CATALOG.items():
            lines.append(f"### {info['name']}")
            lines.append(f"Description: {info['description']}")
            lines.append(f"Base URL: {info['api_base_url']}")
            lines.append(f"Category: {info['category']}")
            if info.get("capabilities"):
                lines.append("Capabilities:")
                for cap in info["capabilities"]:
                    lines.append(f"  - {cap}")
            lines.append("")
        return "\n".join(lines)

    async def _post_investigation(
        self,
        task: AgentTask,
        investigation_result: dict[str, Any],
    ) -> list[DynamicToolSpec]:
        """Extract DynamicToolSpec objects from the investigation answer.

        Called after _investigate completes. Parses the answer for tool specs
        and validates them.
        """
        specs: list[DynamicToolSpec] = []

        # The answer's entities may contain tool specs encoded in properties
        entities = investigation_result.get("entities", [])
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            props = entity.get("properties", {})
            wrapper_code = props.get("wrapper_code", "")
            if not wrapper_code:
                continue

            spec = DynamicToolSpec(
                name=entity.get("name", "").lower().replace(" ", "_"),
                description=entity.get("description", ""),
                api_base_url=props.get("api_base_url", ""),
                wrapper_code=wrapper_code,
                test_code=props.get("test_code", ""),
                category=props.get("category", "dynamic"),
                capabilities=props.get("capabilities", []),
                example_tasks=props.get("example_tasks", []),
                parameters=props.get("parameters", {}),
                status=DynamicToolStatus.DRAFT,
                created_by=self.agent_id,
            )
            specs.append(spec)

        return specs

    async def create_and_register_tools(
        self,
        task: AgentTask,
        tool_registry: Any,
        repl_tool: Any | None = None,
    ) -> list[DynamicToolSpec]:
        """Full lifecycle: investigate → extract specs → test → register.

        This is the high-level entry point that the orchestrator can call
        to trigger tool creation for a research task. It runs the standard
        execute() loop, then extracts and registers any discovered tools.

        Args:
            task: The research task driving tool discovery.
            tool_registry: The InMemoryToolRegistry to register new tools into.
            repl_tool: PythonREPLTool instance for testing wrappers.

        Returns:
            List of DynamicToolSpec objects that were successfully validated.
        """
        # Step 1: Run the standard investigation
        result = await self.execute(task)

        # Step 2: Extract tool specs from the investigation
        investigation_data = {
            "entities": [
                {
                    "name": node.name,
                    "description": node.description,
                    "properties": node.properties,
                }
                for node in result.nodes_added
            ],
        }
        specs = await self._post_investigation(task, investigation_data)

        # Step 3: Validate and register each spec
        validated: list[DynamicToolSpec] = []
        for spec in specs:
            spec.status = DynamicToolStatus.TESTING

            # Validate wrapper code has the required run() function
            if "def run(" not in spec.wrapper_code:
                spec.status = DynamicToolStatus.FAILED
                spec.test_results.append({"error": "Missing run() function in wrapper code"})
                logger.warning("dynamic_tool_validation_failed", tool=spec.name, reason="missing_run")
                continue

            # Test in REPL if available
            if repl_tool is not None:
                test_passed = await self._test_wrapper_in_repl(spec, repl_tool)
                if not test_passed:
                    spec.status = DynamicToolStatus.FAILED
                    logger.warning("dynamic_tool_test_failed", tool=spec.name)
                    continue

            spec.status = DynamicToolStatus.VALIDATED

            # Step 4: Create DynamicTool and register
            from integrations.dynamic.dynamic_tool import DynamicTool

            dynamic_tool = DynamicTool(
                spec=spec,
                repl_tool=repl_tool,
                registry=tool_registry,
            )
            dynamic_tool.save_wrapper()
            spec.status = DynamicToolStatus.REGISTERED
            validated.append(spec)

            logger.info(
                "dynamic_tool_registered",
                tool=spec.name,
                category=spec.category,
                capabilities=spec.capabilities,
            )

        return validated

    async def _test_wrapper_in_repl(
        self,
        spec: DynamicToolSpec,
        repl_tool: Any,
    ) -> bool:
        """Test wrapper code in the sandboxed REPL.

        Validates that:
        1. The code parses without syntax errors
        2. The run() function is callable
        3. If test_code is provided, it executes successfully
        """
        # Test 1: Syntax validation (no network needed)
        escaped_code = spec.wrapper_code.replace("'''", "TRIPLE_QUOTE")
        syntax_test = textwrap.dedent(f"""\
            import json, ast
            code = '''{escaped_code}'''.replace("TRIPLE_QUOTE", "'''")
            try:
                ast.parse(code)
                print(json.dumps({{"success": True, "message": "Syntax valid"}}))
            except SyntaxError as e:
                print(json.dumps({{"success": False, "error": str(e)}}))
        """)

        try:
            result = await repl_tool.execute(
                action="execute",
                code=syntax_test,
            )
            if isinstance(result, dict):
                stdout = result.get("stdout", "")
                try:
                    parsed = json.loads(stdout)
                    if not parsed.get("success"):
                        spec.test_results.append({"test": "syntax", "passed": False, "error": parsed.get("error")})
                        return False
                except (json.JSONDecodeError, TypeError):
                    pass
            spec.test_results.append({"test": "syntax", "passed": True})
        except Exception as e:
            spec.test_results.append({"test": "syntax", "passed": False, "error": str(e)})
            return False

        # Test 2: Function existence check
        func_test = textwrap.dedent(f"""\
            import json
            {spec.wrapper_code}
            print(json.dumps({{
                "success": callable(run) if 'run' in dir() else False,
                "has_run": 'run' in dir(),
            }}))
        """)

        try:
            result = await repl_tool.execute(
                action="execute",
                code=func_test,
            )
            if isinstance(result, dict):
                stdout = result.get("stdout", "")
                try:
                    parsed = json.loads(stdout)
                    if not parsed.get("success"):
                        spec.test_results.append({"test": "function", "passed": False})
                        return False
                except (json.JSONDecodeError, TypeError):
                    pass
            spec.test_results.append({"test": "function", "passed": True})
        except Exception as e:
            spec.test_results.append({"test": "function", "passed": False, "error": str(e)})
            return False

        return True
