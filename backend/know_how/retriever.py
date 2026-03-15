"""LLM-based know-how document retriever.

Selects the most relevant protocol/recipe documents for a given agent task
by asking a fast model (Sonnet) to rank documents based on the task prompt.
Selected documents are injected into the agent system prompt before investigation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import structlog

from core.config import settings

logger = structlog.get_logger(__name__)

# Resolve the know_how directory relative to this file
_KNOW_HOW_DIR = Path(__file__).resolve().parent
_INDEX_PATH = _KNOW_HOW_DIR / "index.json"

# Fast model for retrieval (Sonnet) — cost-efficient, not Opus
_RETRIEVER_MODEL = "claude-sonnet-4-20250514"
_RETRIEVER_MAX_TOKENS = 512


class KnowHowRetriever:
    """Selects relevant know-how documents for a given task using LLM-based ranking."""

    def __init__(self, *, max_docs: int = 3) -> None:
        self.max_docs = max_docs
        self._index = self._load_index()
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    @staticmethod
    def _load_index() -> list[dict[str, Any]]:
        """Load document index from index.json."""
        if not _INDEX_PATH.exists():
            logger.warning("know_how.index_missing", path=str(_INDEX_PATH))
            return []
        with open(_INDEX_PATH) as f:
            data = json.load(f)
        return data.get("documents", [])

    def _build_catalog(self) -> str:
        """Build a compact catalog string for the LLM to rank."""
        lines = []
        for doc in self._index:
            lines.append(
                f"- id: {doc['id']}\n"
                f"  title: {doc['title']}\n"
                f"  description: {doc['description']}\n"
                f"  tags: {', '.join(doc['tags'])}"
            )
        return "\n".join(lines)

    async def retrieve(
        self,
        task_instruction: str,
        agent_type: str = "",
        context: str = "",
    ) -> list[dict[str, Any]]:
        """Select the top-N most relevant documents for a task.

        Uses Sonnet to evaluate relevance and return ranked document IDs.
        Falls back to tag-based matching if the LLM call fails.
        """
        if not self._index:
            return []

        catalog = self._build_catalog()

        prompt = (
            f"You are selecting reference documents for a biomedical research agent.\n\n"
            f"## Agent Type\n{agent_type or 'general'}\n\n"
            f"## Task\n{task_instruction}\n\n"
        )
        if context:
            prompt += f"## Additional Context\n{context}\n\n"

        prompt += (
            f"## Available Documents\n{catalog}\n\n"
            f"## Instructions\n"
            f"Select the {self.max_docs} most relevant documents for this task. "
            f"Return ONLY a JSON array of document IDs in order of relevance, "
            f"most relevant first.\n"
            f"Example: [\"gwas_analysis\", \"pathway_analysis\", \"pandas_bio_recipes\"]\n\n"
            f"If fewer than {self.max_docs} documents are relevant, return fewer. "
            f"If none are relevant, return an empty array []."
        )

        try:
            response = self._client.messages.create(
                model=_RETRIEVER_MODEL,
                max_tokens=_RETRIEVER_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else "[]"
            selected_ids = self._parse_ids(text)
        except Exception as exc:
            logger.warning("know_how.retriever_llm_failed", error=str(exc))
            selected_ids = self._fallback_tag_match(task_instruction)

        # Resolve IDs to full document entries
        id_to_doc = {doc["id"]: doc for doc in self._index}
        return [id_to_doc[doc_id] for doc_id in selected_ids if doc_id in id_to_doc]

    def _parse_ids(self, text: str) -> list[str]:
        """Extract document ID list from LLM response."""
        # Try direct JSON parse
        text = text.strip()
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(x) for x in result[: self.max_docs]]
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from text
        import re

        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return [str(x) for x in result[: self.max_docs]]
            except json.JSONDecodeError:
                pass

        logger.warning("know_how.parse_ids_failed", text=text[:200])
        return []

    def _fallback_tag_match(self, task_instruction: str) -> list[str]:
        """Simple tag-based fallback when LLM is unavailable."""
        task_lower = task_instruction.lower()
        scores: list[tuple[str, int]] = []

        for doc in self._index:
            score = 0
            for tag in doc["tags"]:
                if tag.lower() in task_lower:
                    score += 2
                # Check individual words in multi-word tags
                for word in tag.split("-"):
                    if len(word) > 3 and word in task_lower:
                        score += 1
            # Check title words
            for word in doc["title"].lower().split():
                if len(word) > 3 and word in task_lower:
                    score += 1
            if score > 0:
                scores.append((doc["id"], score))

        scores.sort(key=lambda x: -x[1])
        return [doc_id for doc_id, _ in scores[: self.max_docs]]

    def load_document(self, doc_entry: dict[str, Any]) -> str:
        """Load the full content of a document from disk."""
        doc_path = _KNOW_HOW_DIR / doc_entry["path"]
        if not doc_path.exists():
            logger.warning("know_how.doc_missing", path=str(doc_path))
            return ""
        return doc_path.read_text()

    async def get_context_for_task(
        self,
        task_instruction: str,
        agent_type: str = "",
        context: str = "",
    ) -> str:
        """Main entry point: retrieve and format know-how for injection into system prompt.

        Returns a formatted string ready to be appended to the agent's system prompt.
        """
        docs = await self.retrieve(task_instruction, agent_type, context)
        if not docs:
            return ""

        sections = ["## Domain Know-How (reference protocols and recipes)\n"]
        for doc in docs:
            content = self.load_document(doc)
            if content:
                sections.append(f"### {doc['title']}\n\n{content}")

        return "\n\n".join(sections)
