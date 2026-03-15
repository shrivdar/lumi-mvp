# BioPython Recipes

## Purpose

Common BioPython patterns for sequence manipulation, file parsing, BLAST execution, and biological data retrieval.

## Recipes

### Sequence Manipulation

```python
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Create and manipulate sequences
dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
protein = dna.translate()  # M A I V M G R * K G A R *
complement = dna.complement()
rev_comp = dna.reverse_complement()

# Find ORFs
def find_orfs(seq, min_length=100):
    """Find all ORFs in all 3 reading frames (forward strand)."""
    orfs = []
    for frame in range(3):
        trans = seq[frame:].translate()
        start = 0
        while True:
            start = str(trans).find("M", start)
            if start == -1:
                break
            stop = str(trans).find("*", start)
            if stop == -1:
                stop = len(trans)
            if (stop - start) * 3 >= min_length:
                orfs.append({
                    "frame": frame,
                    "start_aa": start,
                    "end_aa": stop,
                    "length_nt": (stop - start) * 3,
                    "protein": str(trans[start:stop]),
                })
            start = stop + 1
    return orfs
```

### FASTA/GenBank Parsing

```python
from Bio import SeqIO

# Read FASTA
records = list(SeqIO.parse("sequences.fasta", "fasta"))
for rec in records:
    print(f"{rec.id}: {len(rec.seq)} bp")

# Read GenBank — extract CDS features
for rec in SeqIO.parse("genome.gbk", "genbank"):
    for feat in rec.features:
        if feat.type == "CDS":
            gene = feat.qualifiers.get("gene", ["unknown"])[0]
            product = feat.qualifiers.get("product", ["unknown"])[0]
            location = feat.location
            print(f"{gene}: {product} at {location}")

# Write records to FASTA
SeqIO.write(records, "output.fasta", "fasta")

# Convert between formats
SeqIO.convert("input.gbk", "genbank", "output.fasta", "fasta")
```

### BLAST Execution and Parsing

```python
from Bio.Blast import NCBIWWW, NCBIXML

# Run remote BLAST
result_handle = NCBIWWW.qblast("blastp", "nr", str(protein_seq), hitlist_size=10)
blast_records = NCBIXML.parse(result_handle)

for record in blast_records:
    for alignment in record.alignments:
        for hsp in alignment.hsps:
            if hsp.expect < 1e-10:
                print(f"Hit: {alignment.title[:80]}")
                print(f"  E-value: {hsp.expect}")
                print(f"  Identity: {hsp.identities}/{hsp.align_length} "
                      f"({100*hsp.identities/hsp.align_length:.1f}%)")
                print(f"  Score: {hsp.bits}")

# Parse local BLAST XML output
from Bio.Blast import NCBIXML
with open("blast_results.xml") as f:
    for record in NCBIXML.parse(f):
        for hit in record.alignments[:5]:
            best_hsp = hit.hsps[0]
            print(f"{hit.hit_id}: E={best_hsp.expect:.2e}, "
                  f"ID={best_hsp.identities}/{best_hsp.align_length}")
```

### Multiple Sequence Alignment

```python
from Bio.Align.Applications import MafftCommandline

# Run MAFFT alignment
mafft_cline = MafftCommandline(input="sequences.fasta", auto=True)
stdout, stderr = mafft_cline()

# Parse alignment result
from Bio import AlignIO
alignment = AlignIO.read("aligned.fasta", "fasta")
print(f"Alignment: {alignment.get_alignment_length()} columns, {len(alignment)} sequences")

# Calculate conservation per column
from collections import Counter
for i in range(alignment.get_alignment_length()):
    column = alignment[:, i]
    counts = Counter(column)
    most_common = counts.most_common(1)[0]
    conservation = most_common[1] / len(alignment)
    if conservation > 0.9:
        print(f"Position {i}: {most_common[0]} ({conservation:.0%} conserved)")
```

### Entrez Database Queries

```python
from Bio import Entrez

Entrez.email = "your.email@example.com"  # Required by NCBI

# Search PubMed
handle = Entrez.esearch(db="pubmed", term="BRCA1 breast cancer", retmax=10)
record = Entrez.read(handle)
pmids = record["IdList"]

# Fetch abstracts
handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="text")
abstracts = handle.read()

# Search Gene database
handle = Entrez.esearch(db="gene", term="TP53[Gene Name] AND Homo sapiens[Organism]")
record = Entrez.read(handle)

# Fetch protein sequence from GenBank
handle = Entrez.efetch(db="protein", id="NP_000537.3", rettype="fasta", retmode="text")
record = SeqIO.read(handle, "fasta")
```

### PDB Structure Retrieval

```python
from Bio.PDB import PDBList, PDBParser

# Download structure
pdbl = PDBList()
pdbl.retrieve_pdb_file("1TUP", pdir="structures/", file_format="pdb")

# Parse structure
parser = PDBParser(QUIET=True)
structure = parser.get_structure("p53", "structures/pdb1tup.ent")

# Iterate over residues
for model in structure:
    for chain in model:
        for residue in chain:
            if residue.id[0] == " ":  # skip heteroatoms
                ca = residue["CA"] if "CA" in residue else None
                if ca:
                    print(f"Chain {chain.id} Res {residue.get_resname()}{residue.id[1]}: "
                          f"CA at {ca.get_vector()}")
```

## Common Pitfalls

- **Entrez rate limiting**: NCBI limits to 3 requests/second without API key, 10 with. Always set `Entrez.email` and use `Entrez.api_key` if available.
- **BLAST timeout**: Remote BLAST can take minutes. For production use, run local BLAST+ instead.
- **Sequence case sensitivity**: BioPython preserves case; some tools expect uppercase. Use `str(seq).upper()`.
- **GenBank feature parsing**: Qualifier values are lists, not strings. Always index with `[0]`.
- **Memory with large files**: `SeqIO.parse()` is a generator (memory-efficient); `list(SeqIO.parse(...))` loads everything into memory.
