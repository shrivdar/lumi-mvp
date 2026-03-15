"""gnomAD API client — population allele frequencies via GraphQL API."""

from __future__ import annotations

from typing import Any

from core.constants import CACHE_TTL_GNOMAD, RATE_LIMIT_GNOMAD
from integrations.base_tool import BaseTool

GNOMAD_API = "https://gnomad.broadinstitute.org/api"


class GnomADTool(BaseTool):
    tool_id = "gnomad"
    name = "gnomad_search"
    description = "Query gnomAD for population allele frequencies and variant constraint metrics."
    category = "variant"
    rate_limit = RATE_LIMIT_GNOMAD
    cache_ttl = CACHE_TTL_GNOMAD

    async def _execute(self, **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action", "variant")
        if action == "variant":
            return await self._get_variant(
                variant_id=kwargs["variant_id"],
                dataset=kwargs.get("dataset", "gnomad_r4"),
            )
        elif action == "gene":
            return await self._get_gene_constraint(
                gene=kwargs["gene"],
                dataset=kwargs.get("dataset", "gnomad_r4"),
            )
        raise ValueError(f"Unknown gnomAD action: {action}")

    async def _get_variant(self, variant_id: str, dataset: str = "gnomad_r4") -> dict[str, Any]:
        query = """
        query GnomadVariant($variantId: String!, $dataset: DatasetId!) {
          variant(variantId: $variantId, dataset: $dataset) {
            variant_id
            chrom
            pos
            ref
            alt
            exome {
              ac
              an
              af
              populations { id ac an af }
              filters
            }
            genome {
              ac
              an
              af
              populations { id ac an af }
              filters
            }
            rsids
            clinvar_allele_id
            transcript_consequences {
              gene_symbol
              gene_id
              transcript_id
              hgvsc
              hgvsp
              consequence_terms
              lof
              polyphen_prediction
              sift_prediction
            }
          }
        }
        """
        resp = await self._http.post(
            GNOMAD_API,
            json={"query": query, "variables": {"variantId": variant_id, "dataset": dataset}},
        )
        resp.raise_for_status()
        data = resp.json()
        variant = data.get("data", {}).get("variant")
        if not variant:
            return {"variant": None, "variant_id": variant_id}
        return {"variant": self._normalize_variant(variant)}

    async def _get_gene_constraint(self, gene: str, dataset: str = "gnomad_r4") -> dict[str, Any]:
        query = """
        query GnomadGene($gene: String!, $dataset: DatasetId!) {
          gene(gene_symbol: $gene, reference_genome: GRCh38) {
            gene_id
            symbol
            name
            gnomad_constraint {
              exp_lof
              obs_lof
              oe_lof
              oe_lof_lower
              oe_lof_upper
              lof_z
              pLI
              exp_mis
              obs_mis
              oe_mis
              oe_mis_lower
              oe_mis_upper
              mis_z
              exp_syn
              obs_syn
              oe_syn
              syn_z
            }
          }
        }
        """
        resp = await self._http.post(
            GNOMAD_API,
            json={"query": query, "variables": {"gene": gene, "dataset": dataset}},
        )
        resp.raise_for_status()
        data = resp.json()
        gene_data = data.get("data", {}).get("gene")
        if not gene_data:
            return {"gene": None, "query": gene}
        constraint = gene_data.get("gnomad_constraint") or {}
        return {
            "gene": {
                "gene_id": gene_data.get("gene_id", ""),
                "symbol": gene_data.get("symbol", ""),
                "name": gene_data.get("name", ""),
                "constraint": {
                    "pLI": constraint.get("pLI"),
                    "oe_lof": constraint.get("oe_lof"),
                    "oe_lof_lower": constraint.get("oe_lof_lower"),
                    "oe_lof_upper": constraint.get("oe_lof_upper"),
                    "lof_z": constraint.get("lof_z"),
                    "oe_mis": constraint.get("oe_mis"),
                    "mis_z": constraint.get("mis_z"),
                    "oe_syn": constraint.get("oe_syn"),
                    "syn_z": constraint.get("syn_z"),
                },
            },
        }

    @staticmethod
    def _normalize_variant(v: dict[str, Any]) -> dict[str, Any]:
        exome = v.get("exome") or {}
        genome = v.get("genome") or {}
        consequences = v.get("transcript_consequences", [])
        top_consequence = consequences[0] if consequences else {}

        populations = {}
        for pop in (exome.get("populations") or genome.get("populations") or []):
            populations[pop["id"]] = {
                "ac": pop.get("ac", 0),
                "an": pop.get("an", 0),
                "af": pop.get("af", 0),
            }

        return {
            "variant_id": v.get("variant_id", ""),
            "chrom": v.get("chrom", ""),
            "pos": v.get("pos"),
            "ref": v.get("ref", ""),
            "alt": v.get("alt", ""),
            "rsids": v.get("rsids", []),
            "exome_ac": exome.get("ac"),
            "exome_an": exome.get("an"),
            "exome_af": exome.get("af"),
            "genome_ac": genome.get("ac"),
            "genome_an": genome.get("an"),
            "genome_af": genome.get("af"),
            "populations": populations,
            "gene_symbol": top_consequence.get("gene_symbol", ""),
            "consequence": top_consequence.get("consequence_terms", []),
            "hgvsp": top_consequence.get("hgvsp", ""),
            "hgvsc": top_consequence.get("hgvsc", ""),
            "lof": top_consequence.get("lof", ""),
            "clinvar_allele_id": v.get("clinvar_allele_id"),
        }
