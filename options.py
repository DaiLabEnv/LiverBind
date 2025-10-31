import os
import argparse

class Options():
    def __init__(self):
        pass

    def init(self, parser):
        
        # global settings
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        parser.add_argument('--nepoch', type=int, default=200, help='Number of training epochs')
        parser.add_argument('--num_workers', type=int, default=16, help='Number of dataloader workers')
        parser.add_argument('--gpu', type=str, default='0', help='GPUs to use, e.g., "0,1"')
        parser.add_argument('--protein_dim', type=int, default=100, help='Protein embedding dimension')
        parser.add_argument('--atom_dim', type=int, default=34, help='Atom feature dimension')
        parser.add_argument('--protein_feature_dim', type=int, default=70, help='Protein feature dimension')
        parser.add_argument('--hid_dim', type=int, default=64, help='Hidden dimension')
        parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
        parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
        parser.add_argument('--pf_dim', type=int, default=256, help='Feed-forward dimension')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
        parser.add_argument('--kernel_size', type=int, default=9, help='Kernel size for convolutions')
        parser.add_argument('--lr_initial', type=float, default=1e-4, help='Initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

        # args for training
        parser.add_argument('--p_gcn', action='store_true', default=True, help='Use protein GCN')
        parser.add_argument('--m_gcn', action='store_true', default=True, help='Use molecule GCN')
        parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'], help='Task type')
        parser.add_argument('--resume', action='store_true',default=False, help='Resume from pretrained weights')
        parser.add_argument('--pretrain_weights',type=str, default=os.path.join('logs','models','model_best.pth'), help='Path to pretrained weights')
        parser.add_argument('--train_dir', type=str, default=os.path.join('dataset','train_index.csv'), help='Training data CSV')
        parser.add_argument('--test_dir', type=str, default=os.path.join('dataset','test_index.csv'), help='Test data CSV')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./logs/',  help='Directory to save logs and models')
        
        return parser