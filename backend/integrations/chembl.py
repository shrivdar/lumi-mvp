"""ChEMBL API client — drug/compound search, target info, bioactivity data."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_CHEMBL, RATE_LIMIT_CHEMBL
from integrations.base_tool import BaseTool

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


class ChEMBLTool(BaseTool):
    tool_id = "chembl"
    name = "chembl_search"
    description = "Search ChEMBL for drug compounds, targets, mechanisms of action, and bioactivity assays."
    category = "drug"
    rate_limit = RATE_LIMIT_CHEMBL
    cache_ttl = CACHE_TTL_CHEMBL

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "search_compound")
        if action == "search_compound":
            return await self._search_compound(query=kwargs["query"])
        elif action == "compound":
            return await self._get_compound(chembl_id=kwargs["chembl_id"])
        elif action == "target":
            return await self._get_target(target_id=kwargs["target_id"])
        elif action == "search_target":
            return await self._search_target(query=kwargs["query"])
        elif action == "activities":
            return await self._get_activities(chembl_id=kwargs["chembl_id"], max_results=kwargs.get("max_results", 50))
        elif action == "mechanisms":
            return await self._get_mechanisms(chembl_id=kwargs["chembl_id"])
        raise ValueError(f"Unknown ChEMBL action: {action}")

    async def _search_compound(self, query: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{CHEMBL_BASE}/molecule/search.json",
            params={"q": query, "limit": 25},
        )
        resp.raise_for_status()
        data = resp.json()
        compounds = [self._normalize_compound(m) for m in data.get("molecules", [])]
        return {"compounds": compounds, "count": len(compounds), "query": query}

    async def _get_compound(self, chembl_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{CHEMBL_BASE}/molecule/{chembl_id}.json")
        resp.raise_for_status()
        return {"compound": self._normalize_compound(resp.json())}

    async def _search_target(self, query: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{CHEMBL_BASE}/target/search.json",
            params={"q": query, "limit": 25},
        )
        resp.raise_for_status()
        data = resp.json()
        targets = [self._normalize_target(t) for t in data.get("targets", [])]
        return {"targets": targets, "count": len(targets), "query": query}

    async def _get_target(self, target_id: str) -> dict[str, Any]:
        resp = await self._http.get(f"{CHEMBL_BASE}/target/{target_id}.json")
        resp.raise_for_status()
        return {"target": self._normalize_target(resp.json())}

    async def _get_activities(self, chembl_id: str, max_results: int = 50) -> dict[str, Any]:
        resp = await self._http.get(
            f"{CHEMBL_BASE}/activity.json",
            params={"molecule_chembl_id": chembl_id, "limit": min(max_results, 100)},
        )
        resp.raise_for_status()
        data = resp.json()
        activities = []
        for act in data.get("activities", []):
            activities.append({
                "activity_id": act.get("activity_id"),
                "assay_chembl_id": act.get("assay_chembl_id", ""),
                "assay_description": act.get("assay_description", ""),
                "assay_type": act.get("assay_type", ""),
                "target_chembl_id": act.get("target_chembl_id", ""),
                "target_name": act.get("target_pref_name", ""),
                "standard_type": act.get("standard_type", ""),
                "standard_value": act.get("standard_value"),
                "standard_units": act.get("standard_units", ""),
                "standard_relation": act.get("standard_relation", ""),
                "pchembl_value": act.get("pchembl_value"),
            })
        return {"activities": activities, "chembl_id": chembl_id, "count": len(activities)}

    async def _get_mechanisms(self, chembl_id: str) -> dict[str, Any]:
        resp = await self._http.get(
            f"{CHEMBL_BASE}/mechanism.json",
            params={"molecule_chembl_id": chembl_id},
        )
        resp.raise_for_status()
        data = resp.json()
        mechanisms = []
        for mech in data.get("mechanisms", []):
            mechanisms.append({
                "mechanism_of_action": mech.get("mechanism_of_action", ""),
                "action_type": mech.get("action_type", ""),
                "target_chembl_id": mech.get("target_chembl_id", ""),
                "target_name": mech.get("target_name", ""),
                "disease_efficacy": mech.get("disease_efficacy", False),
                "max_phase": mech.get("max_phase"),
            })
        return {"mechanisms": mechanisms, "chembl_id": chembl_id, "count": len(mechanisms)}

    @staticmethod
    def _normalize_compound(mol: dict[str, Any]) -> dict[str, Any]:
        props = mol.get("molecule_properties") or {}
        structs = mol.get("molecule_structures") or {}
        return {
            "chembl_id": mol.get("molecule_chembl_id", ""),
            "name": mol.get("pref_name", ""),
            "molecule_type": mol.get("molecule_type", ""),
            "max_phase": mol.get("max_phase"),
            "first_approval": mol.get("first_approval"),
            "oral": mol.get("oral", False),
            "parenteral": mol.get("parenteral", False),
            "topical": mol.get("topical", False),
            "molecular_weight": props.get("full_mwt"),
            "alogp": props.get("alogp"),
            "hba": props.get("hba"),
            "hbd": props.get("hbd"),
            "psa": props.get("psa"),
            "ro5_violations": props.get("num_ro5_violations"),
            "smiles": structs.get("canonical_smiles", ""),
            "inchi_key": structs.get("standard_inchi_key", ""),
        }

    @staticmethod
    def _normalize_target(target: dict[str, Any]) -> dict[str, Any]:
        components = target.get("target_components", [])
        uniprot_ids = []
        for comp in components:
            for xref in comp.get("target_component_xrefs", []):
                if xref.get("xref_src_db") == "UniProt":
                    uniprot_ids.append(xref.get("xref_id", ""))
        return {
            "target_chembl_id": target.get("target_chembl_id", ""),
            "name": target.get("pref_name", ""),
            "target_type": target.get("target_type", ""),
            "organism": target.get("organism", ""),
            "species_group_flag": target.get("species_group_flag", False),
            "uniprot_ids": uniprot_ids,
        }
