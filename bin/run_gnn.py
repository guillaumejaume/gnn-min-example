#!/usr/bin/env python3
"""
Run model script.
"""
import torch
import argparse
import numpy as np

from core.dataloader.dataloader import make_data_loader
from core.utils import read_params
from core.model import Model


def main(args):

    config_params = read_params(args.config_fpath)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)

    print('*** Create data loader ***')
    dataloader, val_dataloader, test_dataloader = make_data_loader(
        args.batch_size,
        dataset_name='Letter-low',
        cuda=cuda
    )

    print('*** Create model ***')
    model = Model(config=config_params, verbose=True, cuda=cuda)
    if cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # loss function
    loss_fcn = torch.nn.CrossEntropyLoss()

    # Start training
    print('*** Start training ***')
    step = 0
    model.train()
    losses = []
    for epoch in range(args.n_epochs):
        for iter, (graphs, labels) in enumerate(dataloader):

            # forward pass
            logits = model(graphs)

            # compute loss
            loss = loss_fcn(logits, labels)
            losses.append(loss.item())

            # backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # testing
            step += 1
            if step % args.eval_every == 0:
                val_loss, val_acc = test(val_dataloader, model, loss_fcn)
                print("Step {:05d} | Train loss {:.4f} | Over {} | Val loss {:.4f} |"
                      "Val acc {:.4f}".format(step,
                                              np.mean(losses),
                                              len(losses),
                                              val_loss,
                                              val_acc,
                                              ))
                model.train()

    print('*** Start Testing ***')
    test_loss, test_acc = test(test_dataloader, model, loss_fcn)
    print("Test loss {:.4f} | Test acc {:.4f}".format(test_loss, test_acc))


def test(data_loader, model, loss_fcn):
    """
    Testing
    :param data_loader: (data.Dataloader)
    :param model: (Model)
    :param loss_fcn: (torch.nn loss)
    :return: loss, accuracy
    """
    model.eval()
    losses = []
    accuracies = []
    for iter, (graphs, labels) in enumerate(data_loader):

        logits = model(graphs)

        loss = loss_fcn(logits, labels)
        losses.append(loss.item())

        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        accuracies.append(correct.item() * 1.0 / len(labels))

    return np.mean(losses), np.mean(accuracies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Neural Network Minimum Example.')
    parser.add_argument("--config_fpath", type=str, required=True, help="Path to JSON configuration file.")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--eval-every", type=int, default=50, help="eval model every N steps")

    args = parser.parse_args()

    main(args)

