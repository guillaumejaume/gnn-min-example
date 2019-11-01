import torch.utils.data
from dgl.data import tu
import dgl

from core.dataloader.constants import TRAIN_RATIO, TEST_RATIO


def make_data_loader(batch_size, dataset_name='Letter_low', cuda=False):
    """
    Create train/val/test dataloaders
    :param batch_size: batch size (applies for train/test/val)
    :param dataset_name: dataset name, to take from TU dortmund dataset
                         (https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)
    :param cuda: if cuda is available
    :return: train_dataloader, val_dataloader, test_dataloader
    """

    # 1. create train/val/test datasets
    dataset = tu.LegacyTUDataset(name=dataset_name)
    preprocess(dataset, cuda)

    train_size = int(TRAIN_RATIO * len(dataset))
    test_size = int(TEST_RATIO * len(dataset))
    val_size = int(len(dataset) - train_size - test_size)
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset, (train_size, val_size, test_size))

    # 2. create train/val/test dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate
                                                   )

    val_dataloader = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 collate_fn=collate
                                                 )

    test_dataloader = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  collate_fn=collate
                                                  )

    return train_dataloader, val_dataloader, test_dataloader


def collate(data):
    """
    Collate function
    """
    graphs, labels = map(list, zip(*data))
    batched = dgl.batch(graphs)
    labels = torch.LongTensor(labels)
    return batched, labels


def preprocess(dataset, cuda):
    """
    Preprocess graphs by casting into FloatTensor and setting to cuda if available
    :param dataset: (LegacyTUDataset)
    :param cuda: (bool) if cuda is available
    :return:
    """
    for g, _ in dataset:
        for key_g, val_g in g.ndata.items():
            processed = g.ndata.pop(key_g)
            processed = processed.type('torch.FloatTensor')
            if cuda:
                processed = processed.cuda()
            g.ndata[key_g] = processed
        for key_g, val_g in g.edata.items():
            processed = g.edata.pop(key_g)
            processed = processed.type('torch.FloatTensor')
            if cuda:
                processed = processed.cuda()
            g.edata[key_g] = processed
