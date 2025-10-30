import torch
import numpy as np
import pandas as pd
import os
import csv
import subprocess
from Bio import PDB
from Bio.PDB import PDBParser, PPBuilder
from Bio import SeqIO
from Bio.SeqIO import write as seq_write
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

dssp_executable_path = "./bin/dssp"
psiblast_executable_path = "./Blast/ncbi-blast-2.13.0+/bin/psiblast"
psiblast_db_file = "./Blast/ncbi-blast-2.13.0+/database/swissprot"
hhblits_db = "./hh-suite/databases/uniclust30_2018_08/uniclust30_2018_08"

dssp_out_dir = "data/dssp_output"
psiblast_out_dir = "data/psiblast_output"
hhblits_out_dir = "data/hhblits_output"
fasta_out_dir = "data/fasta_output"
protein_feature_dir = "data/protein_information/protein_feature"
contact_map_dir = "data/protein_information/contact_map"

for dir_path in [dssp_out_dir, psiblast_out_dir, hhblits_out_dir,
                 fasta_out_dir, protein_feature_dir, contact_map_dir]:
    os.makedirs(dir_path, exist_ok=True)


def calculate_distance(residue_one, residue_two):
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def generate_contact_map(structure, threshold=8.0):
    model = structure[0]
    residue_list = list(model.get_residues())
    n_residues = len(residue_list)
    contact_map = np.zeros((n_residues, n_residues), dtype=int)

    for i in range(n_residues):
        for j in range(n_residues):
            if i == j:
                contact_map[i][j] = 1
                continue
            try:
                distance = calculate_distance(residue_list[i], residue_list[j])
                if distance < threshold:
                    contact_map[i][j] = 1
            except KeyError:
                pass
    return contact_map


def run_dssp(pdb_file):
    pdbid = os.path.splitext(os.path.basename(pdb_file))[0]
    dssp_output_file = os.path.join(dssp_out_dir, pdbid + ".dssp")
    cmd_args = [dssp_executable_path, "-i", pdb_file, "-o", dssp_output_file]

    try:
        process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error: {stderr.decode('utf-8')}")
        elif not os.path.exists(dssp_output_file):
            print("Error: DSSP output file was not generated.")
        else:
            print("DSSP program ran successfully and generated the output file.")
    except Exception as e:
        print(f"Error: {e}")

    
def run_hhblits(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    ppb = PPBuilder()
    pdbid = os.path.splitext(os.path.basename(pdb_file))[0]
    
    for pp in ppb.build_peptides(structure):
        sequence = pp.get_sequence()
        
    sequence_record = SeqRecord(Seq(sequence), id=pdbid, description="No description")
       
    if not os.path.exists(fasta_out_dir):
        os.mkdir(fasta_out_dir) 
        
    fasta_file = os.path.join(fasta_out_dir, f"{pdbid}.fasta")
    with open(fasta_file, "w") as f:
        seq_write(sequence_record, f, "fasta")
 
    if not os.path.exists(hhblits_out_dir):
        os.mkdir(hhblits_out_dir)
        
    fa_path_list = []
    fa_dir = os.path.dirname(fasta_file)
    new_fa_dir = os.path.join(fa_dir, "tmp_fa_dir")
    os.makedirs(new_fa_dir, exist_ok=True)
    
    with open(fasta_file, 'r') as fasta:
        for record in SeqIO.parse(fasta, 'fasta'):
            record_name = record.name.replace(':', '_')
            single_fasta_path = os.path.join(new_fa_dir, record_name)
            with open(single_fasta_path, 'w') as sf:
                SeqIO.write(record, sf, "fasta")
            fa_path_list.append(single_fasta_path)
            
    for fa in fa_path_list:
        fa_name = os.path.basename(fa)
        out_path = os.path.join(hhblits_out_dir, fa_name)
        os.system(
                f"hhblits -i {fa}"
                f" -d {hhblits_db}"
                f" -cpu 4 -n 4 -e 0.001"
                f" -o {out_path}.hhr -ohhm {out_path}.hhm -oa3m {out_path}.a3m"
            )


def run_psiblast(pdb_file):
    pdbid = os.path.splitext(os.path.basename(pdb_file))[0]
    fasta_file = os.path.join(fasta_out_dir, f"{pdbid}.fasta")
    out_ascii_pssm_file = os.path.join(psiblast_out_dir, pdbid + ".pssm")
    iterations = 3
    cmd_args = [psiblast_executable_path, "-query", fasta_file, "-db", psiblast_db_file, "-num_iterations", str(iterations), "-out_ascii_pssm", out_ascii_pssm_file]

    try:
        if not os.path.exists(psiblast_out_dir):
            os.makedirs(psiblast_out_dir)
        process = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error: {stderr.decode('utf-8')}")
        elif not os.path.exists(out_ascii_pssm_file):
            print("Error: PSSM output file was not generated.")
        else:
            print("PSSM program ran successfully and generated the output file.")
    except Exception as e:
        print(f"Error: {e}")
        

class PDBResidueFeature:
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.res_atom_feas = self.extract_res_atom_feas().float()
        self.dssp = self.extract_dssp().float()
        self.pssm = self.extract_pssm().float()
        self.hhm = self.extract_hhm().float()

    
    def get_pdb_dataframe(self):
        parser = PDBParser()
        structure = parser.get_structure("protein", self.pdb_file)

        data = {"res_id": [], "atom_type": [], "mass": [], "B_factor": [], "is_sidechain": [],
                "charge": [], "num_H": [], "ring": []}
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_id = residue.get_id()[1]
                    for atom in residue:
                        atom_type = atom.element
                        mass = atom.mass
                        B_factor = atom.get_bfactor()
                        is_sidechain = 1 if atom.get_name() not in ["N", "CA", "C", "O"] else 0
                        charge = atom.get_full_id()[4][1]
                        num_H = len([a for a in residue if a.element == "H"])
                        ring = 1 if atom.get_name() in ["CG", "CD", "CE", "CZ"] else 0

                        data["res_id"].append(res_id)
                        data["atom_type"].append(atom_type)
                        data["mass"].append(mass)
                        data["B_factor"].append(B_factor)
                        data["is_sidechain"].append(is_sidechain)
                        data["charge"].append(charge)
                        data["num_H"].append(num_H)
                        data["ring"].append(ring)

        pdb_DF = pd.DataFrame(data)
        res_id_list = pdb_DF["res_id"].unique().tolist()
        return pdb_DF, res_id_list
    
    
    def extract_res_atom_feas(self):
        query_id = os.path.splitext(os.path.basename(self.pdb_file))[0]
        pdb_DF, res_id_list = self.get_pdb_dataframe()
        
        atom_vander_dict = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.85,'H':1.2,'D':1.2,'SE':1.9,'P':1.8,'FE':2.23,'BR':1.95,
                            'F':1.47,'CO':2.23,'V':2.29,'I':1.98,'CL':1.75,'CA':2.81,'B':2.13,'ZN':2.29,'MG':1.73,'NA':2.27,
                            'HG':1.7,'MN':2.24,'K':2.75,'AC':3.08,'AL':2.51,'W':2.39,'NI':2.22}
        for key in atom_vander_dict.keys():
            atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)

        mass = torch.tensor(pdb_DF['mass'].values, dtype=torch.float32).reshape(-1, 1) / 32
        B_factor = torch.tensor(pdb_DF['B_factor'].values, dtype=torch.float32).reshape(-1, 1)
        B_factor_range = torch.max(B_factor) - torch.min(B_factor)
        
        if B_factor_range == 0:
            B_factor = torch.zeros_like(B_factor) + 0.5
        else:
            B_factor = (B_factor - torch.min(B_factor)) / B_factor_range
            
        is_sidechain = torch.tensor(pdb_DF['is_sidechain'].values, dtype=torch.float32).reshape(-1, 1)
        num_H = torch.tensor(pdb_DF['num_H'].values, dtype=torch.float32).reshape(-1, 1)
        ring = torch.tensor(pdb_DF['ring'].values, dtype=torch.float32).reshape(-1, 1)
        atom_type = pdb_DF['atom_type'].tolist()
        atom_vander = torch.zeros((len(atom_type), 1), dtype=torch.float32)
        
        for i, type in enumerate(atom_type):
            try:
                atom_vander[i] = atom_vander_dict[type]
            except:
                atom_vander[i] = atom_vander_dict['C']

            atom_feas = [mass, B_factor, is_sidechain, num_H, ring, atom_vander]
        atom_feas = torch.cat(atom_feas, dim=1)

        res_atom_feas = []
        atom_begin = 0
        
        for res_id in res_id_list:
            res_atom_df = pdb_DF[pdb_DF['res_id'] == res_id]
            atom_num = len(res_atom_df)
            res_atom_feas_i = atom_feas[atom_begin:atom_begin + atom_num]
            res_atom_feas_i = torch.mean(res_atom_feas_i, dim=0).reshape(1, -1)
            res_atom_feas.append(res_atom_feas_i)
            atom_begin += atom_num
        res_atom_feas = torch.cat(res_atom_feas, dim=0)
        
        return res_atom_feas


    def extract_dssp(self):
        maxASA = {'G': 188, 'A': 198, 'V': 220, 'I': 233, 'L': 304, 'F': 272, 'P': 203, 'M': 262, 'W': 317, 'C': 201,
                'S': 234, 'T': 215, 'N': 254, 'Q': 259, 'Y': 304, 'H': 258, 'D': 236, 'E': 262, 'K': 317, 'R': 319}
        map_ss_8 = {' ': [1, 0, 0, 0, 0, 0, 0, 0], 'S': [0, 1, 0, 0, 0, 0, 0, 0], 'T': [0, 0, 1, 0, 0, 0, 0, 0],
                    'H': [0, 0, 0, 1, 0, 0, 0, 0],
                    'G': [0, 0, 0, 0, 1, 0, 0, 0], 'I': [0, 0, 0, 0, 0, 1, 0, 0], 'E': [0, 0, 0, 0, 0, 0, 1, 0],
                    'B': [0, 0, 0, 0, 0, 0, 0, 1]}
        query_id = os.path.splitext(os.path.basename(self.pdb_file))[0]

        query_path  = dssp_out_dir
        dssp_file = os.path.join(query_path, '{}.dssp'.format(query_id))

        with open(dssp_file, 'r') as f:
            text = f.readlines()
        
        start_line = 0
        for i in range(0, len(text)):
            if text[i].startswith('  #  RESIDUE AA STRUCTURE'):
                start_line = i + 1
                break

        dssp = {}
        for i in range(start_line, len(text)):
            line = text[i]
            if line[13] not in maxASA.keys() or line[9] == ' ':
                continue
        
            res_id = float(line[5:10])
            res_dssp = np.zeros([14])
            res_dssp[:8] = map_ss_8[line[16]]
            res_dssp[8] = min(float(line[35:38]) / maxASA[line[13]], 1)
            res_dssp[9] = (float(line[85:91]) + 1) / 2
            res_dssp[10] = min(1, float(line[91:97]) / 180)
            res_dssp[11] = min(1, (float(line[97:103]) + 180) / 360)
            res_dssp[12] = min(1, (float(line[103:109]) + 180) / 360)
            res_dssp[13] = min(1, (float(line[109:115]) + 180) / 360)
            dssp[res_id] = res_dssp.reshape((1, -1))
        
        dssp_tensor = {key : torch.tensor(value, dtype=torch.float32) 
                    for key, value in dssp.items()}
        
        dssp_matrix = np.concatenate(list(dssp.values()), axis=0)

        dssp_tensor = torch.tensor(dssp_matrix, dtype=torch.float32)
        
        return dssp_tensor    


    def extract_pssm(self):
        query_id = os.path.splitext(os.path.basename(self.pdb_file))[0]
        query_path  = psiblast_out_dir
        pssm_file = os.path.join(query_path, '{}.pssm'.format(query_id))
        
        with open(pssm_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content if x.strip()]
        pssm_matrix = []

        for line in content:
            if line[0].isdigit():
                values = line.split()[2:22]
                pssm_row = [int(x) for x in values]
                pssm_matrix.append(pssm_row)
        pssm_matrix = np.array(pssm_matrix)
        
        pssm = 1/(1+np.exp(-pssm_matrix))
        pssm_tensor = torch.tensor(pssm)
        return pssm_tensor


    def extract_hhm(self) :
        query_id = os.path.splitext(os.path.basename(self.pdb_file))[0]
        query_path  = hhblits_out_dir
        hhm_path = os.path.join(query_path, '{}.hhm'.format(query_id))
        hhm_mat = []
        with open(hhm_path, 'r') as mat_file:
            for line in mat_file:
                if line.strip() == '#':
                    break
            for i in range(4):
                mat_file.readline()
            for i, line in enumerate(mat_file):
                if i % 3 == 0:
                    line = line.strip().split()
                    row = list(map(lambda x: 2 ** (-0.001 * float(x)) if x != '*' else 0.0, line[2:-1]))
                elif i % 3 == 1:
                    line = line.strip().split('\t')
                    row.extend(list(map(lambda x: 2 ** (-0.001 * float(x)) if x != '*' else 0.0, line[:-3])))
                    row.extend(list(map(lambda x: float(x) * 0.001 / 20, line[-3:])))
                else:
                    hhm_mat.append(row)
        hhm_tensor = torch.tensor(hhm_mat)
        return hhm_tensor


    def combine_features(self):
        protein_features = torch.cat((self.res_atom_feas, self.dssp, self.pssm, self.hhm), dim=1)
        return protein_features


pdb_folder = 'data/train_protein'
csv_file = './protein_features_shape.csv'


with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['pdbid', 'feature_shape'])

    for root, dirs, files in os.walk(pdb_folder):
        for file in files:
            if file.endswith(".pdb"):
                pdb_file  = os.path.join(root, file)
                pdbid = os.path.splitext(os.path.basename(pdb_file))[0]
                
                dssp_file = os.path.join(dssp_out_dir, f"{pdbid}.dssp")
                if not os.path.isfile(dssp_file):
                    run_dssp(pdb_file)

                hhm_file = os.path.join(hhblits_out_dir, f"{pdbid}.hhm")
                if not os.path.isfile(hhm_file):
                    run_hhblits(pdb_file)

                pssm_file = os.path.join(psiblast_out_dir, f"{pdbid}.pssm")
                if not os.path.isfile(pssm_file):
                    run_psiblast(pdb_file)

                pdb_residue_feature = PDBResidueFeature(pdb_file)

                protein_features = pdb_residue_feature.combine_features()
                
                feature_file = os.path.join(protein_feature_dir, f"protein_feature_{pdbid}.csv")
                
                pd.DataFrame(protein_features.numpy()).to_csv(feature_file, index=False, header=False)
               
                writer.writerow([pdbid, protein_features.shape])

                print(str(pdb_file) + str(protein_features.shape))

parser = PDB.PDBParser(QUIET=True)

for filename in os.listdir(pdb_folder):
    if filename.endswith(".pdb"):
        pdb_path = os.path.join(pdb_folder, filename)
        structure = parser.get_structure("protein", pdb_path)
        
        csv_filename = os.path.splitext(filename)[0] + "_contact_map.csv"
        contact_map_file = os.path.join(contact_map_dir, csv_filename)
        
        if not os.path.exists(contact_map_file): 
            contact_map = generate_contact_map(structure, threshold=8.0)
            print(f'generate_contact_map:{contact_map_file}')
            
            with open(contact_map_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(contact_map)