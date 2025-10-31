import os
import torch
import random
import argparse
import options
import numpy as np
import pandas as pd
import time
import datetime
import timeit
from torch.cuda.amp import GradScaler
from model.model import *
from Dataset import *
from model.gcn_model import *

opt = options.Options().init(argparse.ArgumentParser(description='Protein-small molecule interaction')).parse_args()

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

if __name__ == "__main__":
    log_dir = os.path.join(opt.save_dir)
    result_dir = os.path.join(log_dir, 'results')
    model_dir  = os.path.join(log_dir, 'models')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    logname = os.path.join(result_dir, datetime.datetime.now().isoformat()+'.txt') 

    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
        device = torch.device('cuda:0')
        print('The code uses GPU...', flush=True)
        scaler = GradScaler()
    else:
        device = torch.device('cpu')
        scaler = None
        print('The code uses CPU!!!', flush=True)
        
    dataset_train = []
    data_file = pd.read_csv(opt.train_dir, header=0)
    for i in data_file.values:
        dataset_train.append(i[0])
    dataset_train = shuffle_dataset(dataset_train, 1234)
    dataset_train, dataset_val = split_dataset(dataset_train, 0.8)
    
    task = opt.task
    train_dl = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    val_dl = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    gcn_atom = GCN(nfeat=opt.atom_dim, nhid=opt.atom_dim, nclass=opt.atom_dim, dropout=opt.dropout)
    gcn_protein = GCN(nfeat=opt.protein_feature_dim, nhid=opt.protein_feature_dim, nclass=opt.protein_feature_dim, dropout=opt.dropout)
    encoder_protein = Encoder(opt.protein_dim, opt.hid_dim, opt.n_layers, opt.kernel_size, opt.dropout)
    encoder_atom = Encoder(opt.atom_dim, opt.hid_dim, opt.n_layers, opt.kernel_size, opt.dropout)
    decoder = Decoder(opt.p_gcn, opt.m_gcn, opt.atom_dim, opt.hid_dim, opt.n_layers, opt.n_heads, opt.pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, 
                      opt.dropout, task)
    model = Predictor(encoder_protein, encoder_atom, decoder, gcn_atom, gcn_protein, task)
    if opt.resume:
        path_chk_rest = opt.pretrain_weights 
        print("Resume from "+path_chk_rest)
        from collections import OrderedDict
        state_dict = torch.load(path_chk_rest)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.to(device)
    model = nn.DataParallel(model)
    trainer = Trainer(model, task, lr=opt.lr_initial, weight_decay=opt.weight_decay)
    tester = Tester(model, task)
    epoch_start_time = time.time()

    params = {
          "lr": opt.lr_initial,
          "dropout": opt.dropout,
          "weight_decay": opt.weight_decay,
          "kernel": opt.kernel_size,
          "n_layer": opt.n_layers,
          "batch": opt.batch_size,
          "task": task
          }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_model = os.path.join(model_dir, f"{timestamp}_" + "_".join([f"{k}={v}" for k, v in params.items()]))

    if task == 'classification':
        eval_metrics = ('Epoch\tTime(sec)\tLoss_train\tprecision\tAUROC\tPRAUC')
        with open(logname, 'w') as f:
            f.write(eval_metrics + '\n')

        print('Training...')
        print(eval_metrics)
        start = timeit.default_timer()
        scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=30, gamma=0.5)
        max_AUC_dev = 0
        for epoch in range(1, opt.nepoch + 1):
            epoch_start_time = time.time()
            loss_train = trainer.train(train_dl, epoch, epoch_start_time, device)
            precision, AUROC, PRAUC = tester.test(val_dl, device)
            end = timeit.default_timer()
            time_train = end - start

            metrics = [epoch, time_train, loss_train, precision, AUROC, PRAUC]
            scheduler.step()
            tester.save_metrics(metrics, logname)
            if AUROC > max_AUC_dev:
                tester.save_model(model, file_model)
                max_AUC_dev = AUROC
            print('\t'.join(map(str, metrics)))
            
    elif task == 'regression':
        eval_metrics = ('Epoch\tTime(sec)\tLoss_train\trmse\tr_square\tpear')
        with open(logname, 'w') as f:
            f.write(eval_metrics + '\n')

        print('Training...')
        print(eval_metrics)
        start = timeit.default_timer()
        scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=30, gamma=0.5)
        min_rmse_dev = 10000000
        for epoch in range(1, opt.nepoch + 1):
            epoch_start_time = time.time()
            loss_train = trainer.train(train_dl, epoch, epoch_start_time, device)
            rmse,r_square, pear = tester.test(val_dl, device)
            end = timeit.default_timer()
            time_train = end - start

            metrics = [epoch, time_train, loss_train, rmse, r_square, pear]
            scheduler.step()
            tester.save_metrics(metrics, logname)
            if rmse < min_rmse_dev:
                tester.save_model(model, file_model)
                min_rmse_dev = rmse
            print('\t'.join(map(str, metrics)))