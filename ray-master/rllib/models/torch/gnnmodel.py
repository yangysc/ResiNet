import torch
import torch.nn as nn

from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch import edge_softmax, GATConv
import torch_geometric as tg
from torch.nn import init
import torch_geometric.transforms as T
from dgl.nn.pytorch.conv import GINConv

from torch_geometric.nn.norm import BatchNorm, GraphNorm

from torch_scatter import scatter_add
from torch_geometric.utils import softmax


from torch_geometric.typing import OptTensor

import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.instancenorm import _InstanceNorm
from torch_scatter import scatter
from torch_geometric.utils import degree

from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from torch_geometric.nn.models import JumpingKnowledge
import dgl.function as fn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)





class FALayer(nn.Module):
    def __init__(self, in_dim, dropout):
        super(FALayer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        tem_g = torch.tanh(self.gate(h2)).squeeze()
        e = tem_g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': tem_g}

    def forward(self, g, h):
        g.ndata['h'] = h
        g.apply_edges(self.edge_applying)
        g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return g.ndata['z']


class MessageNorm(nn.Module):
    r"""

    Description
    -----------
    Message normalization was introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    Parameters
    ----------
    learn_scale: bool
        Whether s is a learnable scaling factor or not. Default is False.
    """

    def __init__(self, learn_scale=False):
        super(MessageNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=learn_scale)

    def forward(self, feats, msg, p=2):
        msg = F.normalize(msg, p=2, dim=-1)
        feats_norm = feats.norm(p=p, dim=-1, keepdim=True)
        return msg * feats_norm * self.scale

class FAGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2,learn_msg_scale=False):
        super(FAGCN, self).__init__()
        # self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.msg_norm = torch.nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(hidden_dim, dropout))
            self.norms.append(GraphNormDGL(hidden_dim=hidden_dim))
            self.msg_norm.append(MessageNorm(learn_msg_scale))
        self.t1 = nn.Linear(in_dim, hidden_dim)


        # self.elu = nn.SELU()
        # self.t2 = nn.Linear(hidden_dim, out_dim)
        self.jumpnet = JumpingKnowledge('lstm', channels=hidden_dim, num_layers=2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        # nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, g, h, lg_n_node_valid):

        h = F.dropout(h, p=self.dropout, training=self.training)
        # row-wise normalize node features
        h = h / h.sum(1, keepdim=True).clamp(min=1)
        h = F.normalize(h, p=2, dim=-1)
        h = self.t1(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        emb_node_history = [h]
        for i in range(self.layer_num):
            h1 = self.norms[i](h, lg_n_node_valid)
            h1 = F.selu(h1)
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
            h1 = self.layers[i](g, h1)
            msg = self.msg_norm[i](h, h1)
            h = self.eps * raw + h + msg
            h = F.normalize(h, p=2, dim=-1)

            emb_node_history.append(h)
        # h = self.t2(h)

        h = self.jumpnet(emb_node_history)
        h = F.normalize(h, p=2, dim=-1)
        # y = global_mean_pool(h, g.batch)
        return h




class JumpingKnowledge(torch.nn.Module):
    r"""The Jumping Knowledge layer aggregation module from the
    `"Representation Learning on Graphs with Jumping Knowledge Networks"
    <https://arxiv.org/abs/1806.03536>`_ paper based on either
    **concatenation** (:obj:`"cat"`)

    .. math::

        \mathbf{x}_v^{(1)} \, \Vert \, \ldots \, \Vert \, \mathbf{x}_v^{(T)}

    **max pooling** (:obj:`"max"`)

    .. math::

        \max \left( \mathbf{x}_v^{(1)}, \ldots, \mathbf{x}_v^{(T)} \right)

    or **weighted summation**

    .. math::

        \sum_{t=1}^T \alpha_v^{(t)} \mathbf{x}_v^{(t)}

    with attention scores :math:`\alpha_v^{(t)}` obtained from a bi-directional
    LSTM (:obj:`"lstm"`).

    Args:
        mode (string): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
        channels (int, optional): The number of channels per representation.
            Needs to be only set for LSTM-style aggregation.
            (default: :obj:`None`)
        num_layers (int, optional): The number of layers to aggregate. Needs to
            be only set for LSTM-style aggregation. (default: :obj:`None`)
    """

    def __init__(self, mode, channels=None, num_layers=None):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = nn.GRU(
                channels, (num_layers * channels) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = nn.Linear(2 * ((num_layers * channels) // 2), 1)
            # self.att =  SlimFC(
            #     in_size=2 * ((num_layers * channels) // 2),
            #     out_size=1,
            #     initializer=torch.nn.init.xavier_uniform_,
            #     activation_fn=None)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, xs):
        r"""Aggregates representations across different layers.

        Args:
            xs (list or tuple): List containing layer-wise representations.
        """

        assert isinstance(xs, list) or isinstance(xs, tuple)

        if self.mode == 'cat':
            return torch.cat(xs, dim=-1)
        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'lstm':
            x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)



class EGNet(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, num_heads, num_bases):
        super().__init__()

        aggregators = ['sum', 'mean', 'max']

        # self.encoder = AtomEncoder(hidden_channels)

        self.lin = nn.Linear(input_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        # self.convs.append(EGConv(input_channels, hidden_channels, aggregators,
        #        num_heads, num_bases))
        for _ in range(num_layers):
            self.convs.append(
                EGConv(hidden_channels, hidden_channels, aggregators,
                       num_heads, num_bases))
            self.norms.append(GraphNorm(hidden_dim=hidden_channels))

        # self.mlp =nn.Sequential(
        #     nn.Linear(hidden_channels, hidden_channels // 2, bias=False),
        #     BatchNorm(hidden_channels // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_channels // 2, hidden_channels // 4, bias=False),
        #     BatchNorm(hidden_channels // 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_channels // 4, hidden_channels),
        # )

    def forward(self, data, lg_n_edge_valid):
        tst = T.ToSparseTensor(remove_edge_index=False)
        data_sparse = tst(data)
        x, adj_t, batch = data_sparse.x, data_sparse.adj_t, data.batch
        adj_t = adj_t.set_value(None)  # EGConv works without any edge features

        # x = self.encoder(x)
        x = self.lin(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, adj_t)
            h = norm(h, data.batch, data.n_per_graph)
            h = h.relu_()
            x = x + h

        y = global_mean_pool(x, batch)

        return x, y

class FiLMNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.0):
        super(FiLMNet, self).__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(FiLMConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(FiLMConv(hidden_channels, hidden_channels))
        self.convs.append(FiLMConv(hidden_channels, out_channels, act=None))

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.norms.append(GraphNorm(hidden_dim=hidden_channels))

        self.set2set = Set2Set(hidden_channels, hidden_channels * 2, 10, 2)
        self.lin_g = nn.Linear(hidden_channels * 2, hidden_channels, bias=False)
        self.jumpnet = JumpingKnowledge('lstm', channels=hidden_channels, num_layers=2)

    def forward(self, data):
        x, edge_index, batch, batch_size, n_per_graph = data.x, data.edge_index, data.batch, data.num_graphs, data.n_per_graph
        emb_node_history = []
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index), batch, n_per_graph)
            emb_node_history.append(x)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        emb_node_history.append(x)
        x = self.jumpnet(emb_node_history)
        x = F.normalize(x, p=2, dim=1)
        y = self.set2set(x, batch)
        y = self.lin_g(y)
        return x, y



class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.selu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model

            self.linear = nn.Linear(input_dim, hidden_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append( nn.Linear(hidden_dim, hidden_dim))

            self.linears.append( nn.Linear(hidden_dim, hidden_dim))

            # self.linears.append(SlimFC(
            #         in_size=input_dim,
            #         out_size=hidden_dim,
            #         initializer=torch.nn.init.xavier_uniform_,
            #         activation_fn=None))

            # for layer in range(num_layers - 2):
            #     self.linears.append(SlimFC(
            #         in_size=hidden_dim,
            #         out_size=hidden_dim,
            #         initializer=torch.nn.init.xavier_uniform_,
            #         activation_fn=None))
            # self.linears.append(SlimFC(
            #     in_size=hidden_dim,
            #     out_size=hidden_dim,
            #     initializer=torch.nn.init.xavier_uniform_,
            #     activation_fn=None))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GraphNormDGL(nn.Module):

    def __init__(self, norm_type='gn', hidden_dim=64, print_info=None):
        super(GraphNormDGL, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(hidden_dim),requires_grad=True)
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)

    # batch_list: batch_num_nodes
    def forward(self, tensor, batch_list):
        batch_size = len(batch_list)
        batch_index = torch.arange(batch_size, device=tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:], device=tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        sub = tensor - mean * self.mean_scale
        std = torch.zeros(batch_size, *tensor.shape[1:], device=tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


class GIN(nn.Module):
    """GIN model"""

    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, mode='lstm'):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # self.prelu = nn.PReLU()
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
                self.batch_norms.append(GraphNormDGL(hidden_dim=input_dim))
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
                self.batch_norms.append(GraphNormDGL(hidden_dim=hidden_dim))

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.jumpnet = JumpingKnowledge(mode, channels=output_dim, num_layers=2)
        self.drop = nn.Dropout(final_dropout)

    def forward(self, g, h, n_per_graph):

        hidden_rep = []
        h = self.ginlayers[0](g, h)
        for i in range(1, self.num_layers):
            h = self.ginlayers[i](g, h)
            h = F.selu(h)
            h = self.batch_norms[i](h, n_per_graph)
            h = F.normalize(h, p=2, dim=1)
            hidden_rep.append(h)

        h = self.jumpnet(hidden_rep)
        h = F.normalize(h, p=2, dim=1)

        return h













