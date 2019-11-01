# GNN minimum example

A simple implementation of a Graph Neural Network operating on a directed graph with 
edge weights.

## Installation

### Suggested setup for development

Create a `conda` environment:

```sh
conda env create -f conda.yml
```

Activate it:

```sh
conda activate gnn-min-example
```

Add the root directory to your PYTHONPATH:

```bash
export PYTHONPATH="<YOUR_PATH>/gnn-min-example/"
```

## Code usage:

By default, the GNN is run on the Letter-low dataset. For a comprehensive list of
available TU Dortmund datasets, the reader can refer to this 
[link](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

```sh
python run_gnn.py --config_fpath ../core/config/config_file.json 
```

```bash
usage: run_gnn.py [-h] --config_fpath CONFIG_FPATH [--gpu GPU] [--lr LR]
                  [--weight-decay WEIGHT_DECAY] [--n-epochs N_EPOCHS]
                  [--batch-size BATCH_SIZE] [--eval-every EVAL_EVERY]

Graph Neural Network Minimum Example.

Arguments:
  -h, --help            show this help message and exit
  --config_fpath CONFIG_FPATH
                        Path to JSON configuration file.
  --gpu GPU             gpu (-1 for no GPU, 0 otherwise)
  --lr LR               learning rate
  --weight-decay WEIGHT_DECAY
                        Weight for L2 loss
  --n-epochs N_EPOCHS   number of epochs
  --batch-size BATCH_SIZE
                        batch size
  --eval-every EVAL_EVERY
                        evaluate model every EVAL_EVERY steps
```

The 

## Configuration parameters:

An example configuration file is provided in `core/config/config_files.json`. The 
configuration file is dataset dependent, which means that if the GNN is trained on 
another dataset, some configuration parameters need to be changed accordingly.   

The parameters are:

- `num_layers`: number of GNN layers. Most probably a number between 3 and 5. 
- `node_dim`: input dimension of the nodes. This information is dataset dependent and
 can be found in the `Node Attr. (Dim.)` column
 [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets). 
- `activation`: activation function used between the GNN layers. Choose among
 `'relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu'`. 
- `neighbor_pooling_type`: type of pooling when aggregating node features from the
 neighbors of each node. Choose among `'sum', 'mean''`. 
- Readout parameters:
    - `num_layers`: number of linear layers in the readout function
    - `hidden_dim`: hidden dimension of the readout function
    - `out_dim`: number of classes. This parameter is dataset dependent. 

For more specific information on the GNN parameters, please refer to the implementation
in `core/layers/gnn`. The default configuration file running on the `Letter-low` dataset
is: 


```json
{
  "num_layers": 5,
  "node_dim": 2,
  "activation": "relu",
  "neighbor_pooling_type": "sum",
  "readout": {
      "num_layers": 2,
      "hidden_dim": 64,
      "out_dim": 15
  }
}
```
