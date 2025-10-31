import os
import re
import torch
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from collections import OrderedDict
from gensim.models import Word2Vec

def encode_as_one_hot_vector(element, valid_elements, default=None):
    if element not in valid_elements:
        if default is not None:
            element = default
        else:
            raise ValueError(f"Input element '{element}' is not in the allowed set: {valid_elements}")

    return [int(element == valid_element) for valid_element in valid_elements]


def generate_atom_descriptor(atom):
    atomic_symbols = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']

    bonding_degrees = [0, 1, 2, 3, 4, 5, 6]

    hybridization_states = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 'other'
    ]

    hydrogen_count_options = [0, 1, 2, 3, 4]

    chirality_states = ['R', 'S']

    atom_chirality = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else None

    descriptor_vector = encode_as_one_hot_vector(atom.GetSymbol(), atomic_symbols, default='other') + \
                        encode_as_one_hot_vector(atom.GetDegree(), bonding_degrees) + \
                        encode_as_one_hot_vector(atom.GetHybridization(), hybridization_states, default='other') + \
                        encode_as_one_hot_vector(atom.GetTotalNumHs(), hydrogen_count_options, default=0) + \
                        encode_as_one_hot_vector(atom_chirality, chirality_states, default='R') + \
                        [atom.HasProp('_ChiralityPossible')] + \
                        [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                        [atom.GetIsAromatic()]

    return descriptor_vector


def generate_adjacency_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency,dtype=np.float32)


def extract_molecular_features(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        num_atoms = molecule.GetNumAtoms()
        num_atom_feat = 34
        atom_feature_matrix = np.zeros((num_atoms, num_atom_feat))

        for atom in molecule.GetAtoms():
            atom_feature_matrix[atom.GetIdx(), :] = generate_atom_descriptor(atom)

        atom_adjacency_matrix = generate_adjacency_matrix(molecule)

        return atom_feature_matrix, atom_adjacency_matrix
    except Exception as e:
        raise RuntimeError(f"Error processing SMILES: {e}")


def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens


def generate_token_indices_from_smiles(smiles, target_length=299):
    try:
        token_list = smi_tokenizer(smiles)

        molecule = Chem.MolFromSmiles(smiles)
        if not molecule:
            raise ValueError("Invalid SMILES string: Unable to parse to a molecule.")

        tokens = ['[GLO]'] + token_list
        tokens += ['[PAD]'] * (target_length + 1 - len(tokens))

        valid_tokens = [
        '[PAD]', '[GLO]', 'c', 'C', '(', ')', 'O', '1', '2', '=', 'N', '3', 'n', '4', '[C@H]', 'F',
        '[C@@H]', '-', 'S', '/', 'Cl', '[nH]', 's', 'o', '5', '#', '[C@]', '[C@@]', '\\', '[O-]',
        '[N+]', 'Br', '6', 'P', '[n+]', '7', 'I', '[S+]', '8', '[N-]', '[Si]', 'B', '9', '[2H]',
        '[Se]', '[other_atom]', '[other_token]'
        ]
        token_index_map = {token: idx for idx, token in enumerate(valid_tokens)}

        token_indices = [token_index_map.get(token, token_index_map['[other_token]']) for token in tokens]

        if len(token_indices) != target_length + 1:
            raise ValueError(f"Length of token indices is not {target_length + 1} after processing.")
        return token_indices
    except Exception as e:
        print(f"Error processing SMILES: {smiles} with exception {e}")
        return 0


class ProteinTokenizer:
    padding_token = '<pad>'
    mask_token = '<mask>'
    start_token = class_token = '<cls>'
    end_token = separate_token = '<sep>'
    unknown_token = '<unk>'

    padding_token_id = 0
    mask_token_id = 1
    start_token_id = class_token_id = 2
    end_token_id = separate_token_id = 3
    unknown_token_id = 4

    special_token_ids = [padding_token_id, mask_token_id, start_token_id, end_token_id, unknown_token_id]

    vocab = OrderedDict([
        (padding_token, padding_token_id),
        (mask_token, mask_token_id),
        (class_token, class_token_id),
        (separate_token, end_token_id),
        (unknown_token, unknown_token_id),
        *[(aa, idx + 5) for idx, aa in enumerate("ABCDEFGHIKLMNOPQRSTUVWXYZ")]
    ])

    def tokenize(self, sequence):

        return list(sequence)

    def convert_token_to_id(self, token):

        return self.vocab.get(token, self.unknown_token_id)

    def convert_tokens_to_ids(self, tokens):

        return [self.convert_token_to_id(token) for token in tokens]

    def generate_token_ids(self, sequence, max_length=1000):

        tokens = [self.start_token] + self.tokenize(sequence) + [self.end_token]
        token_ids = self.convert_tokens_to_ids(tokens)
        token_ids += [self.padding_token_id] * (max_length - len(token_ids))
        return token_ids[:max_length]


def load_protein_matrix(unpid):

    feature_file = 'data/protein_information/protein_feature/protein_feature_'+unpid+'.csv'
    adj_file = 'data/protein_information/contact_map/'+unpid+'_contact_map.csv'
    
    protein_feature = pd.read_csv(feature_file, header=None)
    protein_feature = np.array(protein_feature,dtype=np.float32)
    protein_adj = pd.read_csv(adj_file, header=None)
    protein_adj = np.array(protein_adj,dtype=np.int32)
    return protein_feature, protein_adj


def seq_to_kmers_without_overlap(seq, k=3):
    N = len(seq)
    extended_seq = seq[-(k-1):] + seq + seq[:k-1]
    return [extended_seq[i:i+k] for i in range(N)]


def get_protein_embedding(model,protein):
    vec = np.zeros((len(protein), 100))
    i = 0
    for word in protein:
        vec[i, ] = model.wv[word]
        i += 1
    return vec


if __name__ == "__main__":
    
    os.makedirs("dataset", exist_ok=True)
    
    data_path = 'data/train_dataset.csv'

    # Task type: "regression" for continuous values (e.g., affinity), 
    # or "classification" for categorical labels (e.g., active/inactive)
    task = "regression"
    
    with open(data_path,"r") as file: 
        lines = file.readlines()
        header = lines[0]
        data_records = [line.strip() for line in lines[1:]]

    
    num_records = len(data_records)
    protein_tokenizer = ProteinTokenizer()
    model = Word2Vec.load("model/pretrained_word2vec.model")
    interaction_list = []
    i = 0
    
    for index, data in enumerate(data_records):
        print('/'.join(map(str, [index + 1, num_records])))
        smiles, protein, unpid, strength, value = data.strip().split(",")
        smiles_idx = generate_token_indices_from_smiles(smiles)
        atom_feature, atom_adj = extract_molecular_features(smiles)
        protein_feature, protein_adj = load_protein_matrix(unpid)
        protein_embedding = get_protein_embedding(model, seq_to_kmers_without_overlap(protein))
        
        if str(strength) != 'inf':
            atom_feature = torch.FloatTensor(atom_feature)
            atom_adj = torch.FloatTensor(atom_adj)
            smiles_idx = torch.IntTensor(smiles_idx)
            protein_feature = torch.FloatTensor(protein_feature)
            protein_adj = torch.FloatTensor(protein_adj)
            protein = torch.FloatTensor(protein_embedding)
            sequence_idx = protein_tokenizer.generate_token_ids(protein)
            sequence_idx = torch.IntTensor(sequence_idx)

            if task == 'regression':
                label = np.array(value,dtype=np.float32)
                label = torch.FloatTensor(label)

            elif task == 'classification':
                label = int(strength)
                label = torch.tensor(label, dtype=torch.long)
            
            dataset = list([atom_feature, atom_adj, smiles_idx, protein_feature, protein_adj, protein, sequence_idx, label])
            with open("dataset/" + str(i) + '_' + unpid +"_train.txt", "wb") as f:
                pickle.dump(dataset, f)
            interaction_list.append(str(i) + '_' + unpid)
            i += 1
    
    df = pd.DataFrame(data=interaction_list, columns=['interaction_list'])
    df.to_csv('dataset/train_index.csv',sep='\t', index=False, header=True)
    print(f"Processing completed. {len(interaction_list)} entries saved to dataset/")