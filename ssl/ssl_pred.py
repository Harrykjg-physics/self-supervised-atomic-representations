import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tabulate import tabulate

from cgcnn.model import CGCNN
from cgcnn.model import CGCNN_pred
from cgcnn.util import MaskAtom
from cgcnn.dataloader import DataLoaderMasking
from cgcnn.CIFDATA import CIF_Cry_Dataset
from cgcnn.my_logging import *

criterion = nn.CrossEntropyLoss()


def save_checkpoint(state, epoch):
    filename = 'checkingpoint' + '_' + str(epoch) + '_' + '.pth'
    torch.save(state, filename)


def output_training(loss_avg, acc_avg, epoch, log):
    header_1, header_2 = 'Loss | e-stop', 'Acc | TIME'

    train_loss1 = loss_avg['train_loss1'][epoch]
    train_loss2 = loss_avg['train_loss2'][epoch]
    train_loss = loss_avg['train_loss'][epoch]
    train_loss_bond = loss_avg['train_loss_bond'][epoch]

    train_acc1_atom = acc_avg['train_acc1_atom'][epoch]
    train_acc2_atom = acc_avg['train_acc2_atom'][epoch]
    train_acc_atom = acc_avg['train_acc_atom'][epoch]
    train_acc_bond = acc_avg['train_acc_bond'][epoch]

    tab_val = [['Node_prop1', f'{train_loss1:.4f}', f'{train_acc1_atom:.4f}'],
               ['Node_prop2', f'{train_loss2:.4f}', f'{train_acc2_atom:.4f}'],
               ['Node_total', f'{train_loss:.4f}', f'{train_acc_atom:.4f}'],
               ['Bond', f'{train_loss_bond:.4f}', f'{train_acc_bond:.4f}']]

    output = tabulate(tab_val, headers=[f'EPOCH # {epoch}', header_1, header_2], tablefmt='grid_tables')
    print(output)
    log.info("Epoch: {epoch}\t"
             "Node_prop1: {train_acc1_atom}\t"
             "Node_prop2: {train_acc2_atom}\t"
             "Node_total: {train_acc_atom}\t"
             "Bond: {train_acc_bond}\t".format(epoch=epoch,
                                               train_acc1_atom=train_acc1_atom,
                                               train_acc2_atom=train_acc2_atom,
                                               train_acc_atom=train_acc_atom,
                                               train_acc_bond=train_acc_bond))

    return output


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def train(args, model_list, loader, optimizer_list, device, loss_avg, acc_avg):
    model, linear_pred_atoms, linear_pred_bonds = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds = optimizer_list
    # training mode
    model.train()
    linear_pred_atoms.train()
    linear_pred_bonds.train()
    # loss and accuracy for node and edge respectively
    loss1_accum = 0
    acc1_node_accum = 0
    loss2_accum = 0
    acc2_node_accum = 0
    loss_accum = 0
    acc_node_accum = 0
    loss_edge_accum = 0
    acc_edge_accum = 0
    # train over batches
    for step, batch in enumerate(loader):

        batch = batch.to(device)
        # set the atomic vector of masked atoms to zero vector
        batch.x[batch.masked_atom_indices] = torch.zeros([len(batch.masked_atom_indices), batch.x.size()[1]])
        node_rep = model(batch)
        # predicted vector of masked atoms
        pred_node = linear_pred_atoms(node_rep[batch.masked_atom_indices])
        # predict group vector of masked atoms
        pred_node_group_number = pred_node[:, :19]
        # predict period vector of masked atoms
        pred_node_periodic_number = pred_node[:, 19:28]
        # true group number of masked atoms
        true_node_group_number = batch.mask_node_label[:, :19]
        # true period number of masked atoms
        true_node_periodic_number = batch.mask_node_label[:, 19:28]
        true_node_group_number_idx = [list(true_node_group_number[i]).index(1) for i in range(true_node_group_number.size()[0])]
        true_node_group_number_idx = torch.LongTensor(true_node_group_number_idx)
        true_node_periodic_number_idx = [list(true_node_periodic_number[i]).index(1) for i in range(true_node_periodic_number.size()[0])]
        true_node_periodic_number_idx = torch.LongTensor(true_node_periodic_number_idx)
        # calculate the loss of node task
        loss1 = criterion(pred_node_group_number.double(), true_node_group_number_idx)
        loss2 = criterion(pred_node_periodic_number.double(), true_node_periodic_number_idx)
        loss = loss1 + loss2
        # calculate the accuracy of node task
        acc1_node = compute_accuracy(pred_node_group_number, true_node_group_number_idx)
        acc2_node = compute_accuracy(pred_node_periodic_number, true_node_periodic_number_idx)
        acc_node = (acc1_node + acc2_node) / 2
        acc1_node_accum += acc1_node
        acc2_node_accum += acc2_node
        acc_node_accum += acc_node
        # edge task
        if args.mask_edge:
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            # take the sum over connected nodes as the representation of masked edges
            edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            pred_edge = linear_pred_bonds(edge_rep)
            true_edge_idx = []
            for i in range(batch.mask_edge_label.size()[0]):
                max_val = batch.mask_edge_label[i].max()
                max_index = list(batch.mask_edge_label[i]).index(max_val)
                true_edge_idx.append(max_index)
            true_edge_idx = torch.LongTensor(true_edge_idx)
            # calculate the loss of edge task
            loss_edge = criterion(pred_edge.double(), true_edge_idx)
            loss += loss_edge
            loss_edge_accum += loss_edge
            # calculate the accuracy of edge task
            acc_edge = compute_accuracy(pred_edge, true_edge_idx)
            acc_edge_accum += acc_edge

        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_linear_pred_atoms.step()
        optimizer_linear_pred_bonds.step()

        loss1_accum += float(loss1.cpu().item())
        loss2_accum += float(loss2.cpu().item())
        loss_accum += float(loss.cpu().item())
    # average over batches
    loss_avg['train_loss1'].append(loss1_accum / (step + 1))
    loss_avg['train_loss2'].append(loss2_accum / (step + 1))
    loss_avg['train_loss'].append(loss_accum / (step + 1))
    loss_avg['train_loss_bond'].append(loss_edge_accum / (step + 1))
    acc_avg['train_acc1_atom'].append(acc1_node_accum / (step + 1))
    acc_avg['train_acc2_atom'].append(acc2_node_accum / (step + 1))
    acc_avg['train_acc_atom'].append(acc_node_accum / (step + 1))
    acc_avg['train_acc_bond'].append(acc_edge_accum / (step + 1))

    return loss_avg, acc_avg


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of self-supervised training of '
                                                 'graph neural networks')
    parser.add_argument('root', help='The path where you define raw and processed dir')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
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
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=1,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the model')
    parser.add_argument('--model_type', type=str, default="cgcnn", choices=["cgcnn", "gatgnn", "None"])
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataset loading')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device = 'cpu'
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" % (args.num_layer, args.mask_rate, args.mask_edge))

    cwd = os.getcwd()
    log = log_creater(cwd)

    # set up dataset and transform function.
    dataset = CIF_Cry_Dataset(root=args.root,
                              transform=MaskAtom(num_atom_type=92,
                                                 num_edge_type=41,
                                                 mask_rate=args.mask_rate,
                                                 mask_edge=args.mask_edge))

    print("This is length of dataset: ", len(dataset))

    loader = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up models
    if args.model_type == 'cgcnn':
        model = CGCNN(n_conv=args.num_layer,
                      atom_fea_len=args.emb_dim,
                      drop_ratio=args.dropout_ratio).to(device)
        print(model)

    elif args.model_type == 'gatgnn':
        raise NotImplementedError("GATGNN model is not Implemented at present!!!")

    elif args.model_type == "None":
        raise NotImplementedError("Other GNN model is not Implemented at present!!!")

    else:
        raise NotImplementedError("This model is not Implemented !!!")

    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 92).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 41).to(device)

    model_list = [model, linear_pred_atoms, linear_pred_bonds]

    # set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds]

    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        linear_pred_atoms.load_state_dict(checkpoint['linear_pred_atoms_state_dict'])
        linear_pred_bonds.load_state_dict(checkpoint['linear_pred_bonds_state_dict'])
        optimizer_model.load_state_dict(checkpoint['optimizer'])
        optimizer_linear_pred_atoms.load_state_dict(checkpoint['optimizer_linear_pred_atoms'])
        optimizer_linear_pred_bonds.load_state_dict(checkpoint['optimizer_linear_pred_bonds'])

    # define dictionaries to record loss and error
    global loss_avg, acc_avg
    loss_avg = {}
    acc_avg = {}
    loss_avg['train_loss1'] = []
    loss_avg['train_loss2'] = []
    loss_avg['train_loss'] = []
    loss_avg['train_loss_bond'] = []
    acc_avg['train_acc1_atom'] = []
    acc_avg['train_acc2_atom'] = []
    acc_avg['train_acc_atom'] = []
    acc_avg['train_acc_bond'] = []

    # start training loop
    for epoch in range(1, args.epochs + 1):
        # loss_avg and acc_avg save average of loss and accuracy of each epoch
        loss_avg, acc_avg = train(args, model_list, loader, optimizer_list, device, loss_avg, acc_avg)

        output_training(loss_avg, acc_avg, epoch - 1, log)

        if epoch % 5 == 0:
            # torch.save(model.state_dict(), args.output_model_file + '_' + str(epoch) + ".pth")
            save_checkpoint({
                'model_state_dict': model.state_dict(),
                'linear_pred_atoms_state_dict': linear_pred_atoms.state_dict(),
                'linear_pred_bonds_state_dict': linear_pred_bonds.state_dict(),
                'optimizer': optimizer_model.state_dict(),
                'optimizer_linear_pred_atoms': optimizer_linear_pred_atoms.state_dict(),
                'optimizer_linear_pred_bonds': optimizer_linear_pred_bonds.state_dict(),
            }, epoch)

    np.save('loss_avg.npy', loss_avg)
    np.save('acc_avg.npy', acc_avg)


if __name__ == "__main__":
    main()
