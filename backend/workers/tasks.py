"""Celery tasks for async research execution."""

from __future__ import annotations

import asyncio

from workers.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="workers.tasks.run_research",
    max_retries=2,
    default_retry_delay=10,
    acks_late=True,
)
def run_research(self, research_id: str, query: str, config_dict: dict | None = None) -> dict:
    """Run a full research session via the orchestrator.

    Wraps orchestrator.run() and updates session status on completion/failure.
    """
    from core.llm import LLMClient
    from core.models import ResearchConfig, SessionStatus
    from orchestrator.research_loop import ResearchOrchestrator
    from world_model.knowledge_graph import InMemoryKnowledgeGraph

    config = ResearchConfig(**(config_dict or {}))
    kg = InMemoryKnowledgeGraph(graph_id=research_id)
    llm = LLMClient()
    orchestrator = ResearchOrchestrator(llm=llm, kg=kg)

    loop = asyncio.new_event_loop()
    try:
        session = loop.run_until_complete(orchestrator.run(query, config))
        return {
            "research_id": research_id,
            "status": str(session.status),
            "result": session.result.model_dump(mode="json") if session.result else None,
        }
    except Exception as exc:
        try:
            self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {
                "research_id": research_id,
                "status": str(SessionStatus.FAILED),
                "error": str(exc),
            }
    finally:
        loop.close()

    return {"research_id": research_id, "status": "unknown"}


@celery_app.task(
    bind=True,
    name="workers.tasks.run_agent",
    max_retries=2,
    default_retry_delay=5,
    acks_late=True,
)
def run_agent(self, task_dict: dict) -> dict:
    """Run a single agent task (used by orchestrator for Celery dispatch)."""
    from core.models import AgentTask

    task = AgentTask(**task_dict)

    # Agent execution would happen here
    # For now, return a placeholder indicating the task was received
    return {
        "task_id": task.task_id,
        "agent_type": str(task.agent_type),
        "status": "dispatched",
    }
