# RDKit Recipes

## Purpose

Common RDKit patterns for chemical structure analysis, molecular property calculation, similarity searching, and ADMET prediction relevant to drug discovery agents.

## Recipes

### Molecule Creation and Validation

```python
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem

# Create molecule from SMILES
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin
if mol is None:
    raise ValueError("Invalid SMILES")

# Sanitize and add hydrogens
mol = Chem.AddHs(mol)

# Create from InChI
mol = Chem.MolFromInchi("InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)")

# Canonical SMILES (for deduplication)
canonical = Chem.MolToSmiles(mol, canonical=True)
```

### Molecular Property Calculation (Lipinski Rule of 5)

```python
def calculate_properties(mol):
    """Calculate drug-likeness properties."""
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "tpsa": Descriptors.TPSA(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "rings": Descriptors.RingCount(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
    }

def check_lipinski(props):
    """Check Lipinski's Rule of 5 for oral bioavailability."""
    violations = 0
    if props["mw"] > 500: violations += 1
    if props["logp"] > 5: violations += 1
    if props["hbd"] > 5: violations += 1
    if props["hba"] > 10: violations += 1
    return {"violations": violations, "passes": violations <= 1}

# Veber criteria (oral bioavailability)
def check_veber(props):
    return {
        "tpsa_ok": props["tpsa"] <= 140,
        "rotatable_ok": props["rotatable_bonds"] <= 10,
        "passes": props["tpsa"] <= 140 and props["rotatable_bonds"] <= 10,
    }
```

### Molecular Fingerprints and Similarity

```python
from rdkit.Chem import AllChem, DataStructs

# Generate fingerprints
def get_fingerprint(mol, fp_type="morgan"):
    """Generate molecular fingerprint."""
    if fp_type == "morgan":
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    elif fp_type == "rdkit":
        return Chem.RDKFingerprint(mol)
    elif fp_type == "maccs":
        from rdkit.Chem import MACCSkeys
        return MACCSkeys.GenMACCSKeys(mol)

# Tanimoto similarity
def tanimoto_similarity(mol1, mol2, fp_type="morgan"):
    fp1 = get_fingerprint(mol1, fp_type)
    fp2 = get_fingerprint(mol2, fp_type)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Bulk similarity search
def find_similar(query_mol, molecule_list, threshold=0.7):
    """Find molecules similar to query above threshold."""
    query_fp = get_fingerprint(query_mol)
    results = []
    for name, mol in molecule_list:
        fp = get_fingerprint(mol)
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
        if sim >= threshold:
            results.append({"name": name, "similarity": sim})
    return sorted(results, key=lambda x: -x["similarity"])
```

### PAINS Filter (Pan Assay Interference Compounds)

```python
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

def check_pains(mol):
    """Check if molecule contains PAINS substructures."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    entry = catalog.GetFirstMatch(mol)
    if entry is not None:
        return {"is_pains": True, "description": entry.GetDescription()}
    return {"is_pains": False, "description": "Clean"}
```

### Substructure Search

```python
# Substructure matching
query = Chem.MolFromSmarts("[#6](=[#8])-[#7]")  # Amide bond
has_amide = mol.HasSubstructMatch(query)

# Find all matches
matches = mol.GetSubstructMatches(query)
print(f"Found {len(matches)} amide bonds at atoms: {matches}")

# Filter compound library by substructure
def filter_by_substructure(molecules, smarts_pattern):
    """Filter molecules containing a specific substructure."""
    query = Chem.MolFromSmarts(smarts_pattern)
    return [(name, mol) for name, mol in molecules if mol.HasSubstructMatch(query)]
```

### Scaffold Analysis

```python
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(mol):
    """Get Murcko scaffold (core ring system + linkers)."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

def get_generic_scaffold(mol):
    """Get generic scaffold (all atoms replaced with carbon, all bonds single)."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(generic)

# Group compounds by scaffold
from collections import defaultdict
scaffold_groups = defaultdict(list)
for name, mol in molecules:
    scaffold = get_scaffold(mol)
    scaffold_groups[scaffold].append(name)
```

### 3D Conformer Generation

```python
from rdkit.Chem import AllChem

def generate_3d(mol, num_conformers=10):
    """Generate 3D conformers for docking or shape analysis."""
    mol = Chem.AddHs(mol)
    conformer_ids = AllChem.EmbedMultipleConfs(
        mol, numConfs=num_conformers,
        params=AllChem.ETKDGv3()
    )
    # Minimize energy
    results = AllChem.MMFFOptimizeMoleculeConfs(mol)
    energies = [(cid, energy) for cid, (converged, energy) in zip(conformer_ids, results)]
    return mol, sorted(energies, key=lambda x: x[1])
```

### Molecular Descriptors for ML

```python
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd

def compute_descriptor_df(molecules):
    """Compute all RDKit 2D descriptors for a list of molecules."""
    descriptor_names = [name for name, _ in Descriptors.descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    rows = []
    for name, mol in molecules:
        descriptors = calculator.CalcDescriptors(mol)
        rows.append({"name": name, **dict(zip(descriptor_names, descriptors))})
    return pd.DataFrame(rows)
```

## Common Pitfalls

- **SMILES sanitization**: Always check `MolFromSmiles()` return value — it returns `None` for invalid SMILES, not an error.
- **Stereochemistry**: Canonical SMILES may lose stereochemistry info. Use `isomericSmiles=True` in `MolToSmiles()`.
- **Hydrogen handling**: Some operations need explicit Hs (`AddHs`), others need implicit Hs (`RemoveHs`). 3D operations require explicit Hs.
- **Fingerprint radius**: Morgan radius=2 is equivalent to ECFP4 (diameter 4). Radius=3 = ECFP6. Don't confuse radius with diameter.
- **Tanimoto threshold**: Similarity > 0.85 with Morgan FP usually indicates close analogs. Threshold > 0.7 catches broader chemical series.
- **PAINS filters**: PAINS flags are advisory, not definitive — some flagged compounds are genuine drugs. Use as a warning, not a hard filter.
