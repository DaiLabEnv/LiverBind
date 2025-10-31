import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class DTADataset(Dataset):

    def __init__(self, csv_file):
        with open(csv_file,"rb") as f:
            self.datalist = pickle.load(f)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        return data


def packid(batch):
    atoms, adjs, atoms_idxs, protein_features, protein_adjacencies, proteins, protein_idxs, labels = [], [], [], [], [], [], [], []
    for i in batch:
        with open("dataset/" + i + "_train.txt","rb") as f:
            atom_feature, atom_adj, smiles_idx, protein_feature, protein_adj, protein, sequence_idx, label = pickle.load(f)
            atoms.append(atom_feature)
            atoms_idxs.append(smiles_idx)
            protein_features.append(protein_feature)
            protein_adjacencies.append(protein_adj)
            proteins.append(protein)
            protein_idxs.append(sequence_idx)
            labels.append(label)
        
    atoms_len = 300
    proteins_len = 1000

    N = len(atoms)

    atom_num = torch.zeros((N, 1))
    i = 0
    for atom in atoms:
        atom_num[i] = atom.shape[0]
        i += 1
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    protein_num = torch.zeros((N, 1))
    i =0
    for protein in proteins:
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0] + 2

        protein_num[i] = protein.shape[0]
        i += 1

    atoms_new = torch.zeros((N, atoms_len, 34))
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1

    adjs_new = torch.zeros((N, atoms_len, atoms_len))
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1

    proteins_features_new = torch.zeros((N, proteins_len, 70))
    i = 0
    for protein_feature in protein_features:
        p_len = protein_feature.shape[0]
        proteins_features_new[i, :p_len, :] = protein_feature
        i += 1

    protein_adjs_new = torch.zeros((N, proteins_len, proteins_len))
    i = 0
    for protein_adj in protein_adjacencies:
        p_len = protein_adj.shape[0]
        adj = protein_adj
        protein_adjs_new[i, :p_len, :p_len] = adj
        i += 1

    proteins_new = torch.zeros((N, proteins_len, 100))
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        i += 1

    atoms_idxs_new = torch.zeros((N, 300))
    i = 0
    for atoms_idx in atoms_idxs:
        if len(atoms_idx) != 300:
            continue
        else:
            atoms_idxs_new[i, :] = atoms_idx
        i += 1

    protein_idxs_new = torch.zeros((N, proteins_len))
    i = 0
    for protein_idx in protein_idxs:
        prolen = protein_idx.shape[0]
        i += 1
    
    labels_new = torch.zeros(N)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    
    return atoms_new, adjs_new, atoms_idxs_new, proteins_features_new, protein_adjs_new, proteins_new, protein_idxs_new, labels_new, atom_num, protein_num


def collate_fn(batch_identifiers):
    return packid(batch_identifiers)
