# featurize.py
# Takes a SMILES string and turns it into the graph tensors (x, edge_index)
# the models expect. I modeled the atom features on QM9's setup (one-hot atom type + scalars).
# Note: This isn't byte-identical to PyG's built-in QM9 loader (which pulls from precomputed 
# files), but it's close enough for a demo and built on the fly with RDKit.

from typing import Tuple
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# QM9 only has these 5 atoms
ATOM_TYPES = ["H", "C", "N", "O", "F"]

def smiles_to_graph(smiles: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, f"Could not parse '{smiles}' as a valid SMILES string."

    mol = Chem.AddHs(mol)  # QM9 includes explicit hydrogens

    # 3D embedding isn't strictly required for our features, but doesn't hurt
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    except Exception:
        pass 

    if mol.GetNumAtoms() < 2:
        return None, None, "Molecule needs at least 2 atoms to form a graph."
    if mol.GetNumAtoms() > 60:
        return None, None, "Molecule too large for this demo (limit: 60 atoms)."

    atom_features = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        one_hot = [1.0 if symbol == t else 0.0 for t in ATOM_TYPES]
        if sum(one_hot) == 0:
            return None, None, (
                f"Atom type '{symbol}' is outside QM9's covered elements "
                f"({', '.join(ATOM_TYPES)}). Try a molecule with only these atoms."
            )
        extra = [
            float(atom.GetAtomicNum()),
            float(atom.GetIsAromatic()),
            float(atom.GetHybridization() == Chem.HybridizationType.SP),
            float(atom.GetHybridization() == Chem.HybridizationType.SP2),
            float(atom.GetHybridization() == Chem.HybridizationType.SP3),
            float(atom.GetTotalNumHs()),
        ]
        atom_features.append(one_hot + extra)

    x = torch.tensor(atom_features, dtype=torch.float)

    # build edges (both directions)
    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]

    if len(src) == 0:
        return None, None, "Molecule has no bonds - cannot build a graph."

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return x, edge_index, ""