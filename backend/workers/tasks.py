"""Celery tasks for async research execution with DB persistence and checkpoint/resume."""

from __future__ import annotations

import asyncio
import traceback
from typing import Any

import structlog

from workers.celery_app import celery_app

logger = structlog.get_logger(__name__)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine in a new event loop (Celery worker context)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _persist_session_start(
    session_id: str,
    query: str,
    config_dict: dict,
    celery_task_id: str | None = None,
) -> None:
    """Create the research session row in PostgreSQL."""
    from db.persistence import SessionPersistence
    from db.session import async_session_factory

    async with async_session_factory() as db:
        try:
            persistence = SessionPersistence(db)
            await persistence.create_session(
                session_id=session_id,
                query=query,
                config=config_dict,
                celery_task_id=celery_task_id,
            )
            await db.commit()
        except Exception as exc:
            logger.warning("session_persist_start_failed", error=str(exc))


async def _persist_session_end(
    session_id: str,
    status: str,
    result: dict | None = None,
    report_markdown: str | None = None,
    total_tokens: int = 0,
    total_agents: int = 0,
    total_nodes: int = 0,
    total_edges: int = 0,
    total_hypotheses: int = 0,
    current_iteration: int = 0,
) -> None:
    """Update the research session row with final status and result."""
    from db.persistence import SessionPersistence
    from db.session import async_session_factory

    async with async_session_factory() as db:
        try:
            persistence = SessionPersistence(db)
            await persistence.update_session_status(
                session_id,
                status,
                current_iteration=current_iteration,
                total_nodes=total_nodes,
                total_edges=total_edges,
                total_hypotheses=total_hypotheses,
                total_tokens_used=total_tokens,
                total_agents_spawned=total_agents,
            )
            await persistence.complete_session(
                session_id,
                status,
                result=result,
                report_markdown=report_markdown,
            )
            await db.commit()
        except Exception as exc:
            logger.warning("session_persist_end_failed", error=str(exc))


async def _save_checkpoint(
    session_id: str,
    iteration: int,
    orchestrator: Any,
) -> None:
    """Save MCTS checkpoint to PostgreSQL."""
    from db.persistence import SessionPersistence
    from db.session import async_session_factory

    async with async_session_factory() as db:
        try:
            persistence = SessionPersistence(db)

            # Save KG snapshot
            kg_snapshot = orchestrator.kg.to_json()
            kg_snapshot_id = await persistence.save_kg_snapshot(session_id, kg_snapshot)

            # Save hypothesis tree state
            tree_dict = orchestrator.tree.to_dict() if orchestrator.tree else {}

            # Save checkpoint
            await persistence.save_checkpoint(
                session_id=session_id,
                iteration=iteration,
                hypothesis_tree=tree_dict,
                kg_snapshot_id=kg_snapshot_id,
                agent_results=[],  # Results already stored in KG
                session_tokens_used=orchestrator._session_tokens_used,
                total_agents_spawned=orchestrator._total_agents_spawned,
            )

            # Update session stats
            await persistence.update_session_status(
                session_id,
                "RUNNING",
                current_iteration=iteration,
                total_nodes=orchestrator.kg.node_count(),
                total_edges=orchestrator.kg.edge_count(),
                total_hypotheses=orchestrator.tree.node_count if orchestrator.tree else 0,
                total_tokens_used=orchestrator._session_tokens_used,
                total_agents_spawned=orchestrator._total_agents_spawned,
            )

            await db.commit()
        except Exception as exc:
            logger.warning("checkpoint_save_failed", session_id=session_id, error=str(exc))


@celery_app.task(
    bind=True,
    name="workers.tasks.run_research",
    max_retries=2,
    default_retry_delay=10,
    acks_late=True,
    soft_time_limit=7200,  # 2h soft limit
    time_limit=7500,  # 2h 5m hard limit
)
def run_research(self, research_id: str, query: str, config_dict: dict | None = None) -> dict:
    """Run a full research session via the orchestrator.

    Production implementation:
    - Persists session to PostgreSQL at start/end
    - Checkpoints after each MCTS iteration
    - Uses agent factory for real agent execution
    - Generates V2 report
    - Handles failures gracefully with DB status update
    """
    from agents.factory import create_agent
    from core.llm import LLMClient
    from core.models import ResearchConfig, SessionStatus
    from integrations.slack import SlackTool
    from orchestrator.research_loop import ResearchOrchestrator
    from report.generator import generate_report_v2
    from world_model.knowledge_graph import InMemoryKnowledgeGraph

    config = ResearchConfig(**(config_dict or {}))
    kg = InMemoryKnowledgeGraph(graph_id=research_id)
    llm = LLMClient()

    # Build tool instances for dynamic agent tool selection
    tool_instances = _build_tool_instances()

    # Initialize Slack tool for HITL (if configured)
    slack_tool: SlackTool | None = None
    try:
        slack_tool = SlackTool()
    except Exception:
        logger.info("slack_not_configured", research_id=research_id)

    orchestrator = ResearchOrchestrator(
        llm=llm,
        kg=kg,
        agent_factory=create_agent,
        tool_instances=tool_instances,
        slack_tool=slack_tool,
    )

    # Persist session start
    _run_async(_persist_session_start(
        research_id, query, config_dict or {}, self.request.id,
    ))

    try:
        session = _run_async(orchestrator.run(query, config))

        # Generate V2 report
        report_markdown = ""
        try:
            report_markdown = _run_async(generate_report_v2(session, session.result, kg, llm))
        except Exception as report_exc:
            logger.warning("report_v2_generation_failed", error=str(report_exc))
            # Fall back to V1 report
            try:
                from report.generator import generate_report
                report_markdown = _run_async(generate_report(session, session.result, kg, llm))
            except Exception:
                pass

        if session.result:
            session.result.report_markdown = report_markdown

        # Persist final state
        result_dict = session.result.model_dump(mode="json") if session.result else None
        _run_async(_persist_session_end(
            research_id,
            str(session.status),
            result=result_dict,
            report_markdown=report_markdown,
            total_tokens=orchestrator._session_tokens_used,
            total_agents=orchestrator._total_agents_spawned,
            total_nodes=kg.node_count(),
            total_edges=kg.edge_count(),
            total_hypotheses=orchestrator.tree.node_count if orchestrator.tree else 0,
            current_iteration=session.current_iteration,
        ))

        # Save final KG snapshot
        _run_async(_save_checkpoint(research_id, session.current_iteration, orchestrator))

        return {
            "research_id": research_id,
            "status": str(session.status),
            "result": result_dict,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("research_failed", research_id=research_id, error=str(exc), traceback=tb)

        # Persist failure
        _run_async(_persist_session_end(
            research_id,
            str(SessionStatus.FAILED),
            total_tokens=orchestrator._session_tokens_used,
            total_agents=orchestrator._total_agents_spawned,
        ))

        try:
            self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {
                "research_id": research_id,
                "status": str(SessionStatus.FAILED),
                "error": str(exc),
            }

    return {"research_id": research_id, "status": "unknown"}


@celery_app.task(
    bind=True,
    name="workers.tasks.run_agent",
    max_retries=2,
    default_retry_delay=5,
    acks_late=True,
    soft_time_limit=600,  # 10m soft limit
    time_limit=660,  # 11m hard limit
)
def run_agent(self, task_dict: dict) -> dict:
    """Run a single agent task via the agent factory.

    Production implementation:
    - Instantiates agent via factory
    - Executes agent with proper KG, LLM, and tools
    - Returns full AgentResult
    - Isolates failures — never crashes the parent session
    """
    from agents.factory import create_agent
    from core.llm import LLMClient
    from core.models import AgentResult, AgentTask
    from world_model.knowledge_graph import InMemoryKnowledgeGraph

    task = AgentTask(**task_dict)

    try:
        llm = LLMClient()
        # Agent gets a local KG view — merged back by orchestrator
        kg = InMemoryKnowledgeGraph(graph_id=task.research_id)

        tool_instances = _build_tool_instances()

        agent = create_agent(
            agent_type=task.agent_type,
            llm=llm,
            kg=kg,
            tools=tool_instances,
        )

        result: AgentResult = _run_async(agent.execute(task))

        return result.model_dump(mode="json")

    except Exception as exc:
        logger.error(
            "agent_task_failed",
            task_id=task.task_id,
            agent_type=str(task.agent_type),
            error=str(exc),
        )

        try:
            self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            # Return a failure AgentResult instead of crashing
            return AgentResult(
                task_id=task.task_id,
                agent_id=task.agent_id or "unknown",
                agent_type=task.agent_type,
                success=False,
                errors=[str(exc)],
            ).model_dump(mode="json")

    return {
        "task_id": task.task_id,
        "agent_type": str(task.agent_type),
        "status": "unknown",
    }


@celery_app.task(
    bind=True,
    name="workers.tasks.save_checkpoint",
    max_retries=1,
    default_retry_delay=5,
    acks_late=True,
)
def save_checkpoint_task(
    self,
    session_id: str,
    iteration: int,
    hypothesis_tree: dict,
    kg_snapshot: dict,
    session_tokens_used: int,
    total_agents_spawned: int,
) -> dict:
    """Save a checkpoint asynchronously via Celery."""
    async def _save() -> str:
        from db.persistence import SessionPersistence
        from db.session import async_session_factory

        async with async_session_factory() as db:
            persistence = SessionPersistence(db)
            kg_snapshot_id = await persistence.save_kg_snapshot(session_id, kg_snapshot)
            checkpoint_id = await persistence.save_checkpoint(
                session_id=session_id,
                iteration=iteration,
                hypothesis_tree=hypothesis_tree,
                kg_snapshot_id=kg_snapshot_id,
                agent_results=[],
                session_tokens_used=session_tokens_used,
                total_agents_spawned=total_agents_spawned,
            )
            await db.commit()
            return checkpoint_id

    try:
        checkpoint_id = _run_async(_save())
        return {"session_id": session_id, "checkpoint_id": checkpoint_id, "iteration": iteration}
    except Exception as exc:
        logger.warning("checkpoint_task_failed", session_id=session_id, error=str(exc))
        return {"session_id": session_id, "error": str(exc)}


def _build_tool_instances() -> dict[str, Any]:
    """Build tool instances for agent use. Lazy-loads to avoid import cost."""
    tools: dict[str, Any] = {}

    try:
        from integrations.pubmed import PubMedTool
        tools["pubmed"] = PubMedTool()
    except Exception:
        pass

    try:
        from integrations.semantic_scholar import SemanticScholarTool
        tools["semantic_scholar"] = SemanticScholarTool()
    except Exception:
        pass

    try:
        from integrations.uniprot import UniProtTool
        tools["uniprot"] = UniProtTool()
    except Exception:
        pass

    try:
        from integrations.kegg import KEGGTool
        tools["kegg"] = KEGGTool()
    except Exception:
        pass

    try:
        from integrations.reactome import ReactomeTool
        tools["reactome"] = ReactomeTool()
    except Exception:
        pass

    try:
        from integrations.mygene import MyGeneTool
        tools["mygene"] = MyGeneTool()
    except Exception:
        pass

    try:
        from integrations.chembl import ChEMBLTool
        tools["chembl"] = ChEMBLTool()
    except Exception:
        pass

    try:
        from integrations.clinicaltrials import ClinicalTrialsTool
        tools["clinicaltrials"] = ClinicalTrialsTool()
    except Exception:
        pass

    try:
        from integrations.python_repl import PythonREPLTool
        tools["python_repl"] = PythonREPLTool()
    except Exception:
        pass

    return tools
