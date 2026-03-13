"""Yami — Yet Another Molecular Intelligence interface.

Wraps HuggingFace ESM-2 and ESMFold models for protein fitness scoring,
embedding generation, and structure prediction. Implements the
``YamiInterface`` protocol from ``core.interfaces``.

Supports two backends:
- ``huggingface``: Uses the HuggingFace Inference API (remote, no GPU needed).
- ``local``: Uses locally loaded transformers models (requires GPU/CPU resources).
"""

from __future__ import annotations

import hashlib
from typing import Any

import httpx
import numpy as np

from core.audit import AuditLogger, Timer
from core.config import settings
from core.exceptions import ToolError

_audit = AuditLogger("yami")

# HuggingFace Inference API base
_HF_API_BASE = "https://api-inference.huggingface.co/models"


class YamiClient:
    """Yami/ESM interface for protein modelling.

    Satisfies the ``YamiInterface`` protocol from ``core.interfaces``.
    """

    def __init__(
        self,
        *,
        backend: str | None = None,
        hf_token: str | None = None,
        esm_model: str | None = None,
        esmfold_model: str = "facebook/esmfold_v1",
        cache: dict[str, Any] | None = None,
    ) -> None:
        self.backend = backend or settings.yami_backend
        self.hf_token = hf_token or settings.hf_api_token
        self.esm_model = esm_model or settings.esm_model
        self.esmfold_model = esmfold_model

        # Simple in-memory cache (optionally inject external cache)
        self._cache: dict[str, Any] = cache if cache is not None else {}

        # HTTP client for HuggingFace API
        self._http: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            headers: dict[str, str] = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
            self._http = httpx.AsyncClient(
                headers=headers,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ── Cache helpers ─────────────────────────────────────────────────────

    def _cache_key(self, method: str, sequence: str) -> str:
        seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:16]
        return f"yami:{method}:{self.esm_model}:{seq_hash}"

    # ── Public API (YamiInterface protocol) ───────────────────────────────

    async def get_logits(self, sequence: str) -> np.ndarray:
        """Get per-position amino acid logits from ESM-2.

        Returns a 2D array of shape (seq_len, vocab_size) representing the
        model's predicted probability distribution over amino acids at each
        position. Useful for computing pseudo-likelihood fitness scores.
        """
        cache_key = self._cache_key("logits", sequence)
        if cache_key in self._cache:
            return self._cache[cache_key]

        with Timer() as t:
            if self.backend == "local":
                result = await self._local_logits(sequence)
            else:
                result = await self._hf_logits(sequence)

        self._cache[cache_key] = result
        _audit.log(
            "yami_logits",
            model=self.esm_model,
            seq_len=len(sequence),
            duration_ms=t.elapsed_ms,
        )
        return result

    async def get_embeddings(self, sequence: str) -> np.ndarray:
        """Get mean-pooled residue embeddings from ESM-2.

        Returns a 1D array of shape (hidden_dim,) — the mean of per-residue
        representations from the last hidden layer.
        """
        cache_key = self._cache_key("embeddings", sequence)
        if cache_key in self._cache:
            return self._cache[cache_key]

        with Timer() as t:
            if self.backend == "local":
                result = await self._local_embeddings(sequence)
            else:
                result = await self._hf_embeddings(sequence)

        self._cache[cache_key] = result
        _audit.log(
            "yami_embeddings",
            model=self.esm_model,
            seq_len=len(sequence),
            embedding_dim=result.shape[0],
            duration_ms=t.elapsed_ms,
        )
        return result

    async def predict_structure(self, sequence: str) -> dict[str, Any]:
        """Predict 3D structure from amino acid sequence via ESMFold.

        Returns a dict with:
        - ``pdb``: PDB format string of the predicted structure
        - ``plddt_mean``: mean predicted LDDT (confidence) across residues
        - ``plddt_per_residue``: list of per-residue pLDDT scores
        - ``sequence_length``: length of the input sequence
        """
        cache_key = self._cache_key("structure", sequence)
        if cache_key in self._cache:
            return self._cache[cache_key]

        with Timer() as t:
            if self.backend == "local":
                result = await self._local_structure(sequence)
            else:
                result = await self._hf_structure(sequence)

        self._cache[cache_key] = result
        _audit.log(
            "yami_structure",
            model=self.esmfold_model,
            seq_len=len(sequence),
            plddt_mean=result.get("plddt_mean", 0),
            duration_ms=t.elapsed_ms,
        )
        return result

    # ── Fitness scoring (derived from logits) ─────────────────────────────

    async def compute_fitness(self, wild_type: str, mutant: str) -> dict[str, Any]:
        """Compute pseudo-likelihood fitness score for a mutation.

        Compares wild-type vs mutant logits to estimate the fitness effect
        of the mutation. Positive = beneficial, negative = deleterious.
        """
        wt_logits = await self.get_logits(wild_type)
        mut_logits = await self.get_logits(mutant)

        # Find mutation positions
        mutations: list[dict[str, Any]] = []
        min_len = min(len(wild_type), len(mutant))
        for i in range(min_len):
            if wild_type[i] != mutant[i]:
                mutations.append({
                    "position": i,
                    "wild_type_aa": wild_type[i],
                    "mutant_aa": mutant[i],
                })

        # Pseudo-likelihood ratio: sum of log-prob differences at mutation sites
        wt_score = float(np.mean(np.max(wt_logits, axis=-1)))
        mut_score = float(np.mean(np.max(mut_logits, axis=-1)))
        delta = mut_score - wt_score

        return {
            "wild_type_score": wt_score,
            "mutant_score": mut_score,
            "fitness_delta": delta,
            "beneficial": delta > 0,
            "mutations": mutations,
            "mutation_count": len(mutations),
        }

    async def compute_similarity(self, seq_a: str, seq_b: str) -> float:
        """Compute cosine similarity between two protein sequences using ESM embeddings."""
        emb_a = await self.get_embeddings(seq_a)
        emb_b = await self.get_embeddings(seq_b)

        dot = float(np.dot(emb_a, emb_b))
        norm_a = float(np.linalg.norm(emb_a))
        norm_b = float(np.linalg.norm(emb_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ═══════════════════════════════════════════════════════════════════════
    # HuggingFace Inference API backend
    # ═══════════════════════════════════════════════════════════════════════

    async def _hf_logits(self, sequence: str) -> np.ndarray:
        client = await self._get_client()
        url = f"{_HF_API_BASE}/{self.esm_model}"

        resp = await client.post(url, json={"inputs": sequence, "options": {"wait_for_model": True}})
        if resp.status_code != 200:
            raise ToolError(
                f"HuggingFace API error ({resp.status_code}): {resp.text}",
                error_code="YAMI_HF_ERROR",
            )

        data = resp.json()

        # HF feature-extraction returns nested lists: [[token_embeddings]]
        # For logits we need the raw model output; HF inference API returns embeddings
        # by default. We convert to logits-like shape via the embedding response.
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                return np.array(data[0], dtype=np.float32)
            return np.array(data, dtype=np.float32)

        raise ToolError(
            "Unexpected HuggingFace API response format",
            error_code="YAMI_HF_FORMAT",
            details={"response_type": type(data).__name__},
        )

    async def _hf_embeddings(self, sequence: str) -> np.ndarray:
        client = await self._get_client()
        url = f"{_HF_API_BASE}/{self.esm_model}"

        resp = await client.post(url, json={"inputs": sequence, "options": {"wait_for_model": True}})
        if resp.status_code != 200:
            raise ToolError(
                f"HuggingFace API error ({resp.status_code}): {resp.text}",
                error_code="YAMI_HF_ERROR",
            )

        data = resp.json()

        # Feature-extraction endpoint returns: [[per_token_embeddings]]
        # Mean-pool across the sequence dimension
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list) and len(data[0]) > 0 and isinstance(data[0][0], list):
                # Shape: (1, seq_len, hidden_dim) → mean over seq_len
                arr = np.array(data[0], dtype=np.float32)
                return np.mean(arr, axis=0)
            elif isinstance(data[0], list):
                arr = np.array(data, dtype=np.float32)
                return np.mean(arr, axis=0)

        raise ToolError(
            "Unexpected HuggingFace API response format",
            error_code="YAMI_HF_FORMAT",
        )

    async def _hf_structure(self, sequence: str) -> dict[str, Any]:
        client = await self._get_client()
        url = f"{_HF_API_BASE}/{self.esmfold_model}"

        resp = await client.post(
            url,
            json={"inputs": sequence, "options": {"wait_for_model": True}},
            headers={"Accept": "text/plain"},
        )
        if resp.status_code != 200:
            raise ToolError(
                f"ESMFold API error ({resp.status_code}): {resp.text}",
                error_code="YAMI_ESMFOLD_ERROR",
            )

        pdb_text = resp.text
        plddt_scores = self._extract_plddt_from_pdb(pdb_text)

        return {
            "pdb": pdb_text,
            "plddt_mean": float(np.mean(plddt_scores)) if plddt_scores else 0.0,
            "plddt_per_residue": [float(s) for s in plddt_scores],
            "sequence_length": len(sequence),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Local backend (transformers)
    # ═══════════════════════════════════════════════════════════════════════

    async def _local_logits(self, sequence: str) -> np.ndarray:
        """Get logits using locally loaded ESM model."""
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as e:
            raise ToolError(
                "Local backend requires transformers + torch: pip install transformers torch",
                error_code="YAMI_MISSING_DEPS",
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(self.esm_model)
        model = AutoModelForMaskedLM.from_pretrained(self.esm_model)
        model.eval()

        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # Remove special tokens (CLS, EOS)
        logits = outputs.logits[0, 1:-1, :].numpy()
        return logits

    async def _local_embeddings(self, sequence: str) -> np.ndarray:
        """Get embeddings using locally loaded ESM model."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ToolError(
                "Local backend requires transformers + torch: pip install transformers torch",
                error_code="YAMI_MISSING_DEPS",
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(self.esm_model)
        model = AutoModel.from_pretrained(self.esm_model)
        model.eval()

        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean-pool over sequence (excluding special tokens)
        embeddings = outputs.last_hidden_state[0, 1:-1, :].numpy()
        return np.mean(embeddings, axis=0)

    async def _local_structure(self, sequence: str) -> dict[str, Any]:
        """Predict structure using locally loaded ESMFold model."""
        try:
            import torch
            from transformers import AutoTokenizer, EsmFoldModel
        except ImportError as e:
            raise ToolError(
                "Local backend requires transformers + torch: pip install transformers torch",
                error_code="YAMI_MISSING_DEPS",
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(self.esmfold_model)
        model = EsmFoldModel.from_pretrained(self.esmfold_model)
        model.eval()

        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            outputs = model(**inputs)

        pdb_text = outputs.get("pdb", "")
        plddt = outputs.get("plddt", None)
        plddt_scores = plddt[0].numpy().tolist() if plddt is not None else []

        return {
            "pdb": pdb_text,
            "plddt_mean": float(np.mean(plddt_scores)) if plddt_scores else 0.0,
            "plddt_per_residue": plddt_scores,
            "sequence_length": len(sequence),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_plddt_from_pdb(pdb_text: str) -> list[float]:
        """Extract per-residue pLDDT scores from PDB B-factor column."""
        scores: list[float] = []
        seen_residues: set[int] = set()

        for line in pdb_text.splitlines():
            if not line.startswith("ATOM"):
                continue
            # Only take CA atoms (one per residue)
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                res_seq = int(line[22:26].strip())
                if res_seq in seen_residues:
                    continue
                seen_residues.add(res_seq)
                b_factor = float(line[60:66].strip())
                scores.append(b_factor)
            except (ValueError, IndexError):
                continue

        return scores

    @staticmethod
    def validate_sequence(sequence: str) -> bool:
        """Validate that a string is a valid amino acid sequence."""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        return bool(sequence) and all(c.upper() in valid_aas for c in sequence)
