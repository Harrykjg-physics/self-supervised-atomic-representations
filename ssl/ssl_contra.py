import argparse
import os
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from copy import deepcopy

from cgcnn.model import CGCNN
from cgcnn.CIFDATA_cl import CIF_Cry_Dataset
from cgcnn.my_logging import *


def save_checkpoint(state, epoch):
    filename = 'checkingpoint' + '_' + str(epoch) + '.pth'
    torch.save(state, filename)


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x * h, dim=1)


class graphcl(nn.Module):

    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 64))

    def forward_cl(self, batch, batch_batch):
        x = self.gnn(batch)
        x = self.pool(x, batch_batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def train(args, model, device, dataset, optimizer):

    dataset.cell_size = 0

    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.cell_size = args.cell_size1
    dataset2.cell_size = args.cell_size2
    dataset1.aug_ratio = args.aug_ratio1
    dataset2.aug_ratio = args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(zip(loader1, loader2)):
        # print("This is the batch: ", step + 1)
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1, batch1.batch)
        x2 = model.forward_cl(batch2, batch2.batch)
        loss = model.loss_cl(x1, x2)
        # print("This is the loss: ", loss)
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum / (step + 1), train_loss_accum / (step + 1)


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of contrastive learning of '
                                                 'graph neural networks')
    parser.add_argument('root', help='The path where you define raw and processed dir')
    parser.add_argument('--model_type', type=str, default="cgcnn", choices=["cgcnn", "gatgnn"])
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='embedding dimensions (default: 64)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the pre-trained model')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default='none')
    parser.add_argument('--aug_ratio1', type=float, default=0.2)
    parser.add_argument('--aug2', type=str, default='none')
    parser.add_argument('--aug_ratio2', type=float, default=0.2)
    parser.add_argument('--cell_size1', default=0, type=int, choices=[0, 1, 2, 4, 8, 77],
                        help="the augmentations of crytal size")
    parser.add_argument('--cell_size2', default=0, type=int, choices=[0, 1, 2, 4, 8, 77],
                        help="the augmentations of crytal size")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device = 'cpu'
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cwd = os.getcwd()
    log = log_creater(cwd)

    # set up dataset
    dataset = CIF_Cry_Dataset(root=args.root)
    print(dataset)

    # set up model
    if args.model_type == 'cgcnn':
        gnn = CGCNN(n_conv=args.num_layer,
                    atom_fea_len=args.emb_dim,
                    drop_ratio=args.dropout_ratio).to(device)

    elif args.model_type == 'gatgnn':
        print('Not implemented at present')
        assert False
    else:
        print('model error')
        assert False

    model = graphcl(gnn)

    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    global loss_avg, acc_avg
    loss_avg = {}
    acc_avg = {}

    for epoch in range(1, args.epochs):
        print("====epoch " + str(epoch))
        train_acc, train_loss = train(args, model, device, dataset, optimizer)

        loss_avg[epoch] = train_loss
        acc_avg[epoch] = train_acc

        log.info("Epoch: {epoch}\t"
                 "acc: {train_acc}\t"
                 "loss: {train_loss}\t".format(epoch=epoch, train_acc=train_acc, train_loss=train_loss))

        if epoch % 5 == 0:
            save_checkpoint({
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch)

    np.save('loss_avg.npy', loss_avg)
    np.save('acc_avg.npy', acc_avg)


if __name__ == "__main__":
    main()
