"""ESM-2 protein model interface — embeddings, fitness predictions via HuggingFace Inference API."""

from __future__ import annotations

from typing import Any

from core.config import settings
from core.constants import CACHE_TTL_ESM, RATE_LIMIT_ESM
from core.exceptions import ToolError
from integrations.base_tool import BaseTool

HF_INFERENCE_BASE = "https://api-inference.huggingface.co/models"


class ESMTool(BaseTool):
    tool_id = "esm"
    name = "esm_predict"
    description = (
        "Run ESM-2 protein language model for embeddings, fitness predictions, and masked token prediction."
    )
    category = "protein"
    rate_limit = RATE_LIMIT_ESM
    cache_ttl = CACHE_TTL_ESM
    timeout = 120  # protein model inference can be slow

    @property
    def _model(self) -> str:
        return settings.esm_model

    @property
    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if settings.hf_api_token:
            h["Authorization"] = f"Bearer {settings.hf_api_token}"
        return h

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "embeddings")
        if action == "embeddings":
            return await self._get_embeddings(sequence=kwargs["sequence"])
        elif action == "fill_mask":
            return await self._fill_mask(sequence=kwargs["sequence"])
        elif action == "fitness":
            return await self._predict_fitness(
                sequence=kwargs["sequence"],
                mutations=kwargs.get("mutations", []),
            )
        raise ValueError(f"Unknown ESM action: {action}")

    async def _get_embeddings(self, sequence: str) -> dict[str, Any]:
        self._validate_sequence(sequence)
        resp = await self._http.post(
            f"{HF_INFERENCE_BASE}/{self._model}",
            json={"inputs": sequence, "options": {"wait_for_model": True}},
            headers=self._headers,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # HF returns list of embeddings per token
        return {
            "sequence": sequence,
            "length": len(sequence),
            "model": self._model,
            "embeddings": data,
        }

    async def _fill_mask(self, sequence: str) -> dict[str, Any]:
        if "<mask>" not in sequence:
            raise ToolError(
                "Sequence must contain <mask> token for fill_mask action",
                error_code="ESM_INVALID_INPUT",
            )
        resp = await self._http.post(
            f"{HF_INFERENCE_BASE}/{self._model}",
            json={"inputs": sequence, "options": {"wait_for_model": True}},
            headers=self._headers,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        predictions: list[dict[str, Any]] = []
        if isinstance(data, list):
            for pred in data[:20]:
                predictions.append({
                    "token": pred.get("token_str", ""),
                    "score": pred.get("score", 0.0),
                    "sequence": pred.get("sequence", ""),
                })
        return {
            "sequence": sequence,
            "model": self._model,
            "predictions": predictions,
        }

    async def _predict_fitness(
        self, sequence: str, mutations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Predict fitness by comparing wild-type and mutant log-likelihoods via masked marginals."""
        self._validate_sequence(sequence)
        if not mutations:
            return {"sequence": sequence, "model": self._model, "fitness_scores": []}

        scores: list[dict[str, Any]] = []
        for mutation in mutations:
            # Parse mutation string like "A123V" → pos=122, wt=A, mut=V
            if len(mutation) < 3:
                continue
            wt_aa = mutation[0]
            mut_aa = mutation[-1]
            try:
                pos = int(mutation[1:-1]) - 1  # 0-indexed
            except ValueError:
                continue

            if pos < 0 or pos >= len(sequence):
                continue

            # Create masked sequence
            masked = sequence[:pos] + "<mask>" + sequence[pos + 1:]
            resp = await self._http.post(
                f"{HF_INFERENCE_BASE}/{self._model}",
                json={"inputs": masked, "options": {"wait_for_model": True}},
                headers=self._headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            wt_score = 0.0
            mut_score = 0.0
            if isinstance(data, list):
                for pred in data:
                    token = pred.get("token_str", "").strip()
                    if token == wt_aa:
                        wt_score = pred.get("score", 0.0)
                    if token == mut_aa:
                        mut_score = pred.get("score", 0.0)

            scores.append({
                "mutation": mutation,
                "position": pos + 1,
                "wt_residue": wt_aa,
                "mut_residue": mut_aa,
                "wt_score": wt_score,
                "mut_score": mut_score,
                "delta_score": mut_score - wt_score,
                "predicted_effect": "beneficial" if mut_score > wt_score else "deleterious",
            })

        return {
            "sequence": sequence[:50] + "..." if len(sequence) > 50 else sequence,
            "model": self._model,
            "fitness_scores": scores,
        }

    @staticmethod
    def _validate_sequence(sequence: str) -> None:
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = set(sequence.upper()) - valid_aa
        if invalid:
            raise ToolError(
                f"Invalid amino acid characters: {invalid}",
                error_code="ESM_INVALID_SEQUENCE",
            )
