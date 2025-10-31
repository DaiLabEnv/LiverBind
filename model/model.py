import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, average_precision_score, mean_squared_error, r2_score
from scipy.stats import pearsonr
from torch.optim.optimizer import Optimizer

def to_cuda(data, device='cuda:0', cuda_available=True):
    compound, adj, atoms_idx, protein_feature, protein_adj, protein, protein_idx, correct_interaction, atom_num, protein_num = data

    if cuda_available:
        compound = compound.to(device)
        adj = adj.to(device)
        protein = protein.to(device)
        protein_feature = protein_feature.to(device)
        protein_adj = protein_adj.to(device)     
        atom_num = atom_num.to(device)
        protein_num = protein_num.to(device)
        correct_interaction = correct_interaction.to(device)
        atoms_idx = atoms_idx.to(device)
        protein_idx = protein_idx.to(device)

    return compound, adj, atoms_idx, protein_feature, protein_adj, protein, protein_idx, correct_interaction, atom_num, protein_num

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(K.device)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        x = self.fc(x)

        return x


class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            scale = self.scale.to(protein.device)
            conved = (conved + conv_input) * scale
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved



class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)
        
    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.do(F.relu(self.fc_1(x)))

        x = self.fc_2(x)

        x = x.permute(0, 2, 1)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout)
        self.ea = self_attention(hid_dim, n_heads, dropout)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    def __init__(self, p_gcn, m_gcn, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, task):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.p_gcn = p_gcn
        self.m_gcn = m_gcn
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.sa = self_attention(hid_dim, n_heads, dropout)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.ft2 = nn.Linear(70, hid_dim)
        self.ft_p = nn.Linear(34, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        if task == 'classification':
            self.fc_2 = nn.Linear(256, 2)
        elif task == 'regression':
            self.fc_2 = nn.Linear(256, 1)
        self.gn_1 = nn.GroupNorm(4, 256)
        self.weight1 = nn.Parameter(torch.FloatTensor(64, 64))
        self.weight2 = nn.Parameter(torch.FloatTensor(64, 64))

        self.weight3 = nn.Parameter(torch.FloatTensor(64, 64))
        self.weight4 = nn.Parameter(torch.FloatTensor(64, 64))

    def forward(self, trg1, trg2, src1, src2, trg_mask=None,src_mask=None):
        if self.m_gcn == True:
            trg1 = self.ft(trg1)
            trg = torch.matmul(trg1, self.weight1) + torch.matmul(trg2, self.weight2)
        else:
            trg = self.ft(trg1)
        
        if self.p_gcn == True:
            src1 = self.ft2(src1)
            src = torch.matmul(src1, self.weight3) + torch.matmul(src2, self.weight4)
        else:
            src = src2

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)
        sum = torch.zeros((trg.shape[0], self.hid_dim)).to(trg.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j, ]
                v = v * norm[i, j]
                sum[i, ] += v
        label = self.gn_1(F.relu(self.fc_1(sum)))
        label = self.fc_2(label)
        return label


class Predictor(nn.Module):
    def __init__(self, encoder_protein, encoder_atom, decoder, gcn_atom, gcn_protein, task, atom_dim=34):
        super().__init__()

        self.encoder_protein = encoder_protein
        self.encoder_atom = encoder_atom
        self.decoder = decoder
        self.gcn_atom = gcn_atom
        self.gcn_protein = gcn_protein
        self.task = task
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def gcn(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj, support)
        return output

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num) 
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :int(atom_num[i][0])] = 1
            protein_mask[i, :int(protein_num[i][0])] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        return compound_mask, protein_mask


    def forward(self, compound, adj, atoms_idx, protein_feature, protein_adj, protein, protein_idx, atom_num, protein_num):
        compound_max_len = compound.shape[1]
        protein_max_len = protein.shape[1]
        compound_mask, protein_mask = self.make_masks(atom_num, protein_num, compound_max_len, protein_max_len)
        compound_mask, protein_mask = compound_mask.to(compound.device), protein_mask.to(compound.device)
        compound_gcn = self.gcn_atom(compound, adj)
        protein_gcn = self.gcn_protein(protein_feature, protein_adj)
        enc_src = self.encoder_protein(protein)
        enc_tag = self.encoder_atom(compound)
        out = self.decoder(compound_gcn, enc_tag, protein_gcn, enc_src, compound_mask, protein_mask)
        return out

    def __call__(self, data, train=True):
        compound, adj, atoms_idx, protein_feature, protein_adj, protein, protein_idx, correct_interaction, atom_num, protein_num = data
        
        if self.task == 'classification':
            Loss = nn.CrossEntropyLoss()
        elif self.task == 'regression':
            Loss = nn.MSELoss(reduction='mean')

        if train:
            predicted_interaction = self.forward(compound, adj, atoms_idx, protein_feature, protein_adj, protein, protein_idx, atom_num, protein_num)
            loss = Loss(predicted_interaction, correct_interaction.view(-1, 1))
            return loss

        else:
            predicted_interaction = self.forward(compound, adj, atoms_idx, protein_feature, protein_adj, protein, protein_idx, atom_num, protein_num)
            return correct_interaction, predicted_interaction


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])

                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size)

                p.data.copy_(p_data_fp32)

        return loss
    
    
def pack(atoms, adjs, proteins, labels, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        i += 1
    labels_new = torch.zeros(N, dtype=torch.float, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return (atoms_new, adjs_new, proteins_new, labels_new, atom_num, protein_num)


class Trainer(object):
    def __init__(self, model, task, lr, weight_decay, scaler=None):
        self.model = model
        self.scaler = scaler
        weight_p, bias_p = [], []

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)

    
    def train(self, dataloader, epoch, epoch_start_time, device):
        self.model.train()
        loss_train = 0
        if self.scaler is None:
            for i, data_pack in enumerate(tqdm(dataloader), 0):
                data_pack = to_cuda(data_pack, device=device)
                loss = self.model(data_pack)

                self.optimizer.zero_grad()
                try:
                    loss.backward(torch.ones_like(loss)/4)
                    self.optimizer.step()

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of GPU memory, skipping batch')
                        torch.cuda.empty_cache()
                    else:
                        print(e)

                loss_train += loss.detach().mean().item()
                print("------------------------------------------------------------------")
                print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, time.time()-epoch_start_time, loss.detach().mean().item()))
                print("------------------------------------------------------------------")
        
        else:
            for i, data_pack in enumerate(tqdm(dataloader), 0):
                data_pack = to_cuda(data_pack, device=device)

                loss = self.model(data_pack)

                self.optimizer.zero_grad()
                self.scaler.scale(loss.mean()).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_train += loss.detach().mean().item()
                print("------------------------------------------------------------------")
                print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, time.time()-epoch_start_time, loss.detach().mean().item()))
                print("------------------------------------------------------------------")
        return loss.detach().sum().item()


class Tester(object):
    def __init__(self, model, task):
        self.model = model
        self.task = task

    def test(self, dataset, device):
        self.model.eval()
        if self.task == 'classification':
            N = len(dataset)
            T, S, P = [], [], []
            with torch.no_grad():
                for i, data_pack in enumerate(tqdm(dataset), 0):
                    data_pack = to_cuda(data_pack, device=device)
                    correct_labels, predicted_interaction = self.model(data=data_pack, train=False)
                    correct_labels = correct_labels.to('cpu').data.numpy()
                    predicted_interaction = predicted_interaction.to('cpu').numpy()
                    if predicted_interaction.shape[1] == 2:
                        predicted_interaction = predicted_interaction[:, 1]
                    else:
                        predicted_interaction = predicted_interaction.squeeze()
                    
                    threshold = 0.5
                    predicted_labels = (predicted_interaction >= threshold).astype(int)

                    correct_labels = np.ravel(correct_labels)
                    predicted_interaction = np.ravel(predicted_interaction)
                    
                    T.extend(correct_labels)
                    S.extend(predicted_interaction)
                    P.extend(predicted_labels)
            T = np.array(T)
            S = np.array(S)
            P = np.array(P)
            precision = precision_score(T, P)
            AUROC = roc_auc_score(T, S)
            PRAUC = average_precision_score(T, S)

            return precision, AUROC, PRAUC, 

        elif self.task == 'regression':
            T, S = [], []
            threshold=5.
            with torch.no_grad():
                for i, data_pack in enumerate(tqdm(dataset), 0):
                    data_pack = to_cuda(data_pack, device=device)
                    correct_labels, predicted_interaction = self.model(data=data_pack, train=False)
                    correct_labels = correct_labels.to('cpu').data.numpy()
                    predicted_interaction = predicted_interaction.to('cpu').data.numpy()
                    predicted_scores = predicted_interaction[:,0]
                    T.extend(correct_labels)
                    S.extend(predicted_scores)
            T = np.array(T)
            S = np.array(S)

            rmse = mean_squared_error(T, S, squared=False)
            r_square = r2_score(T, S)
            pearson_corr = pearsonr(T, S)[0]

            return rmse, r_square, pearson_corr

    def save_metrics(self, metric, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, metric)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)