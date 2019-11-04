import torch
import torch.nn as nn
import dgl

from core.layers.gnn import GINLayer
from core.layers.mlp import MLP
from core.layers.constants import GNN_NODE_FEAT_IN


class Model(nn.Module):

    def __init__(self, config, verbose=False, cuda=False):
        """
        Model Constructor
        :param config: (dict) configurations to build the GNN extractor
        :param verbose: (bool) verbosity level
        :param cuda: (bool) if cuda is available
        """
        super(Model, self).__init__()
        self.verbose = verbose
        self.cuda = cuda

        # gnn parameters
        self.num_layers = config['num_layers']
        self.node_dim = config['node_dim']
        self.act = config['activation']
        self.neighbor_pooling_type = config['neighbor_pooling_type']

        if verbose:
            print('Creating GNN layers...')
        self._build_gnn()

        # readout parameters
        self.readout_num_layers = config['readout']['num_layers']
        self.readout_h_dim = config['readout']['hidden_dim']
        self.readout_out_dim = config['readout']['out_dim']

        if verbose:
            print('Creating readout function...')

        self._build_readout()

    def _build_gnn(self):
        """
        Build the GNN layers
        """
        self.gnn = nn.ModuleList()
        for i in range(self.num_layers):
            self.gnn.append(
                GINLayer(
                    node_dim=self.node_dim,
                    hidden_dim=self.node_dim,
                    out_dim=self.node_dim,
                    act=self.act,
                    neighbor_pooling_type=self.neighbor_pooling_type,
                    layer_id=i,
                    verbose=self.verbose))

    def _build_readout(self):
        """
        Build the graph readout
        """
        in_dim = self.node_dim * self.num_layers
        self.readout = MLP(
            in_dim,
            self.readout_h_dim,
            self.readout_out_dim,
            self.readout_num_layers,
            verbose=self.verbose)

    def forward(self, graphs):
        """
        Model forward
        :param graphs: (DGL batch) batch of DGLGraphs
        :return: graph embedding
        """
        assert graphs.number_of_nodes() > 1, "Number of nodes in a graph must be >1."

        bs = graphs.batch_size

        for i, layer in enumerate(self.gnn):

            # forward-pass
            h = layer(graphs)

            # update node features
            graphs.ndata[GNN_NODE_FEAT_IN] = h

            # store h
            key = 'h_' + str(i)
            graphs.ndata[key] = h

        h = torch.stack([dgl.sum_nodes(graphs, 'h_' + str(i))
                         for i in range(self.num_layers)], dim=1).view(bs, -1)
        graph_embeddings = self.readout(h)

        if self.cuda:
            graph_embeddings = graph_embeddings.cuda()

        return graph_embeddings
