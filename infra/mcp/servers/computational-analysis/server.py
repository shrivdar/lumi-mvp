"""Computational Analysis MCP Server — local scientific computations.

Provides cheminformatics (RDKit), sequence analysis (BioPython), and
statistical tools (SciPy/scikit-learn) that run entirely locally without
external API calls.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from base_server import APIClient, Server, error_result, make_tool, start, text_result

import json
import math
from typing import Any

import numpy as np
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
from scipy import stats
from sklearn.decomposition import PCA

server = Server("computational-analysis")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string into an RDKit Mol, returning None on failure."""
    return Chem.MolFromSmiles(smiles)


def _compute_properties(mol: Chem.Mol, smiles: str) -> dict[str, Any]:
    """Compute a standard panel of molecular properties."""
    return {
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 3),
        "exact_mass": round(Descriptors.ExactMolWt(mol), 5),
        "logp": round(Descriptors.MolLogP(mol), 3),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "ring_count": Lipinski.RingCount(mol),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "fraction_csp3": round(Descriptors.FractionCSP3(mol), 3),
        "lipinski_violations": sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            Descriptors.NumHDonors(mol) > 5,
            Descriptors.NumHAcceptors(mol) > 10,
        ]),
    }


def _morgan_fp(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> DataStructs.ExplicitBitVect:
    """Generate a Morgan (circular) fingerprint."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list:
    return [
        # -- Cheminformatics --------------------------------------------------
        make_tool(
            "smiles_to_properties",
            "Compute molecular properties (MW, LogP, TPSA, HBD/HBA, Lipinski, etc.) from a SMILES string.",
            {
                "smiles": {"type": "string", "description": "SMILES string of the molecule"},
            },
            required=["smiles"],
        ),
        make_tool(
            "smiles_fingerprint",
            "Compute a Morgan (circular) fingerprint from a SMILES string. Returns bit indices that are set.",
            {
                "smiles": {"type": "string", "description": "SMILES string of the molecule"},
                "radius": {"type": "integer", "description": "Fingerprint radius (default 2)", "default": 2},
                "n_bits": {"type": "integer", "description": "Fingerprint length in bits (default 2048)", "default": 2048},
            },
            required=["smiles"],
        ),
        make_tool(
            "smiles_similarity",
            "Compute Tanimoto similarity between two molecules given their SMILES strings.",
            {
                "smiles_a": {"type": "string", "description": "SMILES of first molecule"},
                "smiles_b": {"type": "string", "description": "SMILES of second molecule"},
                "radius": {"type": "integer", "description": "Morgan FP radius (default 2)", "default": 2},
                "n_bits": {"type": "integer", "description": "Fingerprint length (default 2048)", "default": 2048},
            },
            required=["smiles_a", "smiles_b"],
        ),
        make_tool(
            "smiles_substructure",
            "Check whether a query substructure (SMARTS or SMILES) is present in a molecule.",
            {
                "smiles": {"type": "string", "description": "SMILES of the molecule to search in"},
                "query": {"type": "string", "description": "SMARTS or SMILES pattern to search for"},
            },
            required=["smiles", "query"],
        ),
        # -- Sequence analysis -------------------------------------------------
        make_tool(
            "sequence_analysis",
            "Basic nucleotide sequence analysis: length, GC content, base composition, dinucleotide frequencies.",
            {
                "sequence": {"type": "string", "description": "DNA or RNA sequence (ACGTU)"},
            },
            required=["sequence"],
        ),
        make_tool(
            "sequence_translate",
            "Translate a DNA coding sequence to a protein sequence using the standard codon table.",
            {
                "sequence": {"type": "string", "description": "DNA coding sequence (length should be a multiple of 3)"},
                "frame": {"type": "integer", "description": "Reading frame offset: 0, 1, or 2 (default 0)", "default": 0},
            },
            required=["sequence"],
        ),
        make_tool(
            "sequence_reverse_complement",
            "Return the reverse complement of a DNA sequence.",
            {
                "sequence": {"type": "string", "description": "DNA sequence (ACGT)"},
            },
            required=["sequence"],
        ),
        # -- Statistics --------------------------------------------------------
        make_tool(
            "statistical_test",
            "Run a statistical test. Supported: ttest_ind, ttest_paired, mannwhitney, fisher_exact, chi2, ks_2samp.",
            {
                "test": {
                    "type": "string",
                    "description": "Test name: ttest_ind, ttest_paired, mannwhitney, fisher_exact, chi2, ks_2samp",
                },
                "group_a": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "First data group (numeric array). For fisher_exact, provide the 2x2 table as [a, b, c, d].",
                },
                "group_b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Second data group (numeric array). Not used for fisher_exact or chi2.",
                },
                "alternative": {
                    "type": "string",
                    "description": "Alternative hypothesis: two-sided, less, greater (default two-sided)",
                    "default": "two-sided",
                },
            },
            required=["test", "group_a"],
        ),
        make_tool(
            "enrichment_analysis",
            "Hypergeometric enrichment test (e.g., gene-set over-representation). Returns p-value and fold enrichment.",
            {
                "drawn_successes": {"type": "integer", "description": "Number of successes in the drawn sample (k)"},
                "sample_size": {"type": "integer", "description": "Total number drawn (n)"},
                "population_successes": {"type": "integer", "description": "Total successes in population (K)"},
                "population_size": {"type": "integer", "description": "Total population size (N)"},
            },
            required=["drawn_successes", "sample_size", "population_successes", "population_size"],
        ),
        make_tool(
            "pca_analysis",
            "Run PCA on a numeric matrix. Returns explained variance and principal components.",
            {
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "2D numeric matrix (list of rows). Rows = samples, columns = features.",
                },
                "n_components": {
                    "type": "integer",
                    "description": "Number of principal components to return (default min(5, n_features, n_samples))",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional row labels (sample names)",
                },
            },
            required=["matrix"],
        ),
    ]


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    try:
        # -- Cheminformatics tools ----------------------------------------

        if name == "smiles_to_properties":
            smiles = arguments["smiles"]
            mol = _mol_from_smiles(smiles)
            if mol is None:
                return error_result(f"Invalid SMILES: {smiles}")
            return text_result(_compute_properties(mol, smiles))

        elif name == "smiles_fingerprint":
            smiles = arguments["smiles"]
            radius = arguments.get("radius", 2)
            n_bits = arguments.get("n_bits", 2048)
            mol = _mol_from_smiles(smiles)
            if mol is None:
                return error_result(f"Invalid SMILES: {smiles}")
            fp = _morgan_fp(mol, radius=radius, n_bits=n_bits)
            on_bits = list(fp.GetOnBits())
            return text_result({
                "smiles": smiles,
                "radius": radius,
                "n_bits": n_bits,
                "num_on_bits": len(on_bits),
                "on_bits": on_bits,
                "density": round(len(on_bits) / n_bits, 4),
            })

        elif name == "smiles_similarity":
            smiles_a = arguments["smiles_a"]
            smiles_b = arguments["smiles_b"]
            radius = arguments.get("radius", 2)
            n_bits = arguments.get("n_bits", 2048)
            mol_a = _mol_from_smiles(smiles_a)
            mol_b = _mol_from_smiles(smiles_b)
            if mol_a is None:
                return error_result(f"Invalid SMILES (molecule A): {smiles_a}")
            if mol_b is None:
                return error_result(f"Invalid SMILES (molecule B): {smiles_b}")
            fp_a = _morgan_fp(mol_a, radius=radius, n_bits=n_bits)
            fp_b = _morgan_fp(mol_b, radius=radius, n_bits=n_bits)
            tanimoto = DataStructs.TanimotoSimilarity(fp_a, fp_b)
            dice = DataStructs.DiceSimilarity(fp_a, fp_b)
            return text_result({
                "smiles_a": smiles_a,
                "smiles_b": smiles_b,
                "tanimoto_similarity": round(tanimoto, 6),
                "dice_similarity": round(dice, 6),
                "radius": radius,
                "n_bits": n_bits,
            })

        elif name == "smiles_substructure":
            smiles = arguments["smiles"]
            query = arguments["query"]
            mol = _mol_from_smiles(smiles)
            if mol is None:
                return error_result(f"Invalid SMILES: {smiles}")
            # Try SMARTS first, fall back to SMILES
            pattern = Chem.MolFromSmarts(query)
            if pattern is None:
                pattern = Chem.MolFromSmiles(query)
            if pattern is None:
                return error_result(f"Invalid query pattern: {query}")
            has_match = mol.HasSubstructMatch(pattern)
            matches = mol.GetSubstructMatches(pattern)
            return text_result({
                "smiles": smiles,
                "query": query,
                "match": has_match,
                "num_matches": len(matches),
                "atom_indices": [list(m) for m in matches],
            })

        # -- Sequence analysis tools --------------------------------------

        elif name == "sequence_analysis":
            raw_seq = arguments["sequence"].upper().replace(" ", "").replace("\n", "")
            seq = Seq(raw_seq)
            length = len(seq)
            if length == 0:
                return error_result("Empty sequence provided")

            # Base composition
            composition = {}
            for base in "ACGTU":
                count = raw_seq.count(base)
                if count > 0:
                    composition[base] = count

            # GC content
            gc = round(gc_fraction(seq) * 100, 2)

            # Dinucleotide frequencies
            dinucs: dict[str, int] = {}
            for i in range(length - 1):
                di = raw_seq[i : i + 2]
                dinucs[di] = dinucs.get(di, 0) + 1

            # Sort dinucleotides by frequency descending
            sorted_dinucs = dict(sorted(dinucs.items(), key=lambda x: x[1], reverse=True))

            is_rna = "U" in raw_seq and "T" not in raw_seq
            return text_result({
                "length": length,
                "type": "RNA" if is_rna else "DNA",
                "gc_content_percent": gc,
                "base_composition": composition,
                "dinucleotide_frequencies": sorted_dinucs,
            })

        elif name == "sequence_translate":
            raw_seq = arguments["sequence"].upper().replace(" ", "").replace("\n", "")
            frame = arguments.get("frame", 0)
            if frame not in (0, 1, 2):
                return error_result("Frame must be 0, 1, or 2")
            # Convert RNA to DNA for codon lookup
            dna = raw_seq.replace("U", "T")
            dna = dna[frame:]
            # Trim to multiple of 3
            trim_len = (len(dna) // 3) * 3
            dna = dna[:trim_len]
            if len(dna) == 0:
                return error_result("Sequence too short for translation after applying frame offset")

            protein = []
            codons_used = []
            for i in range(0, len(dna), 3):
                codon = dna[i : i + 3]
                aa = CODON_TABLE.get(codon, "X")
                protein.append(aa)
                codons_used.append(codon)

            protein_str = "".join(protein)
            # Trim trailing stop
            if protein_str.endswith("*"):
                protein_str = protein_str[:-1]

            return text_result({
                "protein_sequence": protein_str,
                "length_nt": len(dna),
                "length_aa": len(protein_str),
                "frame": frame,
                "num_codons": len(codons_used),
                "has_start_codon": codons_used[0] == "ATG" if codons_used else False,
                "has_stop_codon": protein[-1] == "*" if protein else False,
            })

        elif name == "sequence_reverse_complement":
            raw_seq = arguments["sequence"].upper().replace(" ", "").replace("\n", "")
            seq = Seq(raw_seq)
            rc = str(seq.reverse_complement())
            return text_result({
                "input_sequence": raw_seq,
                "reverse_complement": rc,
                "length": len(rc),
            })

        # -- Statistical tools --------------------------------------------

        elif name == "statistical_test":
            test = arguments["test"]
            group_a = np.array(arguments["group_a"], dtype=float)
            group_b_raw = arguments.get("group_b")
            group_b = np.array(group_b_raw, dtype=float) if group_b_raw else None
            alternative = arguments.get("alternative", "two-sided")

            if test == "ttest_ind":
                if group_b is None:
                    return error_result("ttest_ind requires group_b")
                stat, p = stats.ttest_ind(group_a, group_b, alternative=alternative)
                effect_size = (np.mean(group_a) - np.mean(group_b)) / np.sqrt(
                    (np.var(group_a, ddof=1) + np.var(group_b, ddof=1)) / 2
                )  # Cohen's d
                return text_result({
                    "test": "independent_t_test",
                    "statistic": round(float(stat), 6),
                    "p_value": float(p),
                    "mean_a": round(float(np.mean(group_a)), 6),
                    "mean_b": round(float(np.mean(group_b)), 6),
                    "cohens_d": round(float(effect_size), 6),
                    "n_a": len(group_a),
                    "n_b": len(group_b),
                    "alternative": alternative,
                })

            elif test == "ttest_paired":
                if group_b is None:
                    return error_result("ttest_paired requires group_b")
                if len(group_a) != len(group_b):
                    return error_result("Paired t-test requires equal-length groups")
                stat, p = stats.ttest_rel(group_a, group_b, alternative=alternative)
                diffs = group_a - group_b
                effect_size = float(np.mean(diffs) / np.std(diffs, ddof=1))
                return text_result({
                    "test": "paired_t_test",
                    "statistic": round(float(stat), 6),
                    "p_value": float(p),
                    "mean_diff": round(float(np.mean(diffs)), 6),
                    "cohens_d": round(effect_size, 6),
                    "n": len(group_a),
                    "alternative": alternative,
                })

            elif test == "mannwhitney":
                if group_b is None:
                    return error_result("mannwhitney requires group_b")
                stat, p = stats.mannwhitneyu(group_a, group_b, alternative=alternative)
                # Rank-biserial correlation as effect size
                n1, n2 = len(group_a), len(group_b)
                r = 1 - (2 * stat) / (n1 * n2)
                return text_result({
                    "test": "mann_whitney_u",
                    "statistic": round(float(stat), 6),
                    "p_value": float(p),
                    "rank_biserial_r": round(float(r), 6),
                    "n_a": n1,
                    "n_b": n2,
                    "alternative": alternative,
                })

            elif test == "fisher_exact":
                table_flat = arguments["group_a"]
                if len(table_flat) != 4:
                    return error_result("fisher_exact requires group_a as [a, b, c, d] for the 2x2 table")
                table = [[int(table_flat[0]), int(table_flat[1])],
                         [int(table_flat[2]), int(table_flat[3])]]
                odds_ratio, p = stats.fisher_exact(table, alternative=alternative)
                return text_result({
                    "test": "fisher_exact",
                    "odds_ratio": round(float(odds_ratio), 6),
                    "p_value": float(p),
                    "contingency_table": table,
                    "alternative": alternative,
                })

            elif test == "chi2":
                # group_a is the observed contingency table flattened row-major;
                # if group_b is provided it's the shape [nrows, ncols], otherwise assume 2x(len/2)
                data = arguments["group_a"]
                if group_b is not None and len(group_b) == 2:
                    nrows, ncols = int(group_b[0]), int(group_b[1])
                else:
                    ncols = 2
                    nrows = len(data) // ncols
                if nrows * ncols != len(data):
                    return error_result(f"Cannot reshape {len(data)} values into {nrows}x{ncols} table")
                observed = np.array(data, dtype=float).reshape(nrows, ncols)
                chi2, p, dof, expected = stats.chi2_contingency(observed)
                return text_result({
                    "test": "chi_squared",
                    "statistic": round(float(chi2), 6),
                    "p_value": float(p),
                    "degrees_of_freedom": int(dof),
                    "expected": expected.round(3).tolist(),
                })

            elif test == "ks_2samp":
                if group_b is None:
                    return error_result("ks_2samp requires group_b")
                stat, p = stats.ks_2samp(group_a, group_b, alternative=alternative)
                return text_result({
                    "test": "kolmogorov_smirnov_2sample",
                    "statistic": round(float(stat), 6),
                    "p_value": float(p),
                    "n_a": len(group_a),
                    "n_b": len(group_b),
                    "alternative": alternative,
                })

            else:
                return error_result(
                    f"Unknown test: {test}. Supported: ttest_ind, ttest_paired, mannwhitney, fisher_exact, chi2, ks_2samp"
                )

        elif name == "enrichment_analysis":
            k = int(arguments["drawn_successes"])
            n = int(arguments["sample_size"])
            K = int(arguments["population_successes"])
            N = int(arguments["population_size"])

            if n > N or k > n or k > K or K > N:
                return error_result(
                    f"Invalid parameters: k={k}, n={n}, K={K}, N={N}. "
                    "Must satisfy k<=n<=N and k<=K<=N."
                )

            # P(X >= k) — upper tail for over-representation
            p_value = float(stats.hypergeom.sf(k - 1, N, K, n))
            # Expected overlap
            expected = (n * K) / N if N > 0 else 0
            fold_enrichment = k / expected if expected > 0 else float("inf")

            return text_result({
                "test": "hypergeometric_enrichment",
                "drawn_successes_k": k,
                "sample_size_n": n,
                "population_successes_K": K,
                "population_size_N": N,
                "p_value": p_value,
                "expected_overlap": round(expected, 4),
                "fold_enrichment": round(fold_enrichment, 4),
                "significant_005": p_value < 0.05,
                "significant_001": p_value < 0.01,
            })

        elif name == "pca_analysis":
            matrix = np.array(arguments["matrix"], dtype=float)
            if matrix.ndim != 2:
                return error_result("Matrix must be 2-dimensional")
            n_samples, n_features = matrix.shape
            if n_samples < 2:
                return error_result("Need at least 2 samples (rows) for PCA")
            max_components = min(n_samples, n_features)
            n_components = arguments.get("n_components", min(5, max_components))
            n_components = min(n_components, max_components)
            labels = arguments.get("labels")

            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(matrix)

            components_out = []
            for i in range(n_components):
                comp = {
                    "pc": i + 1,
                    "explained_variance_ratio": round(float(pca.explained_variance_ratio_[i]), 6),
                    "explained_variance": round(float(pca.explained_variance_[i]), 6),
                    "loadings": np.round(pca.components_[i], 6).tolist(),
                }
                components_out.append(comp)

            # Build sample projections
            projections = []
            for i in range(n_samples):
                proj: dict[str, Any] = {}
                if labels and i < len(labels):
                    proj["label"] = labels[i]
                for j in range(n_components):
                    proj[f"PC{j + 1}"] = round(float(transformed[i, j]), 6)
                projections.append(proj)

            return text_result({
                "n_samples": n_samples,
                "n_features": n_features,
                "n_components": n_components,
                "total_explained_variance": round(float(sum(pca.explained_variance_ratio_)), 6),
                "components": components_out,
                "projections": projections,
            })

        return error_result(f"Unknown tool: {name}")

    except Exception as exc:
        return error_result(f"{name} failed: {exc}")


if __name__ == "__main__":
    start(server)
