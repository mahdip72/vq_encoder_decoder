import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_scatter import scatter_mean
import torch, functools
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,global_mean_pool, global_max_pool
from torch_scatter import scatter_add
from . import GVP, GVPConvLayer, LayerNorm, tuple_index

class structure_encoder(nn.Module):
    '''
    GVP-GNN for protein sequence representation, inspired by gearnet/gvp.py
    modified based on GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers, out_dims
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''

    def __init__(self, node_in_dim, node_h_dim,
                 edge_in_dim, edge_h_dim,
                 seq_in=False, seq_embed_mode="embedding", seq_embed_dim=20,
                 num_layers=3, drop_rate=0.1,
                 vector_gate=True,
                 ):

        super(structure_encoder, self).__init__()
        self.seq_embed_mode = seq_embed_mode
        if seq_in and seq_embed_mode == "embedding":
            self.W_s = nn.Embedding(20, seq_embed_dim)
            node_in_dim = (node_in_dim[0] + seq_embed_dim, node_in_dim[1])
        elif seq_in:
            node_in_dim = (node_in_dim[0] + seq_embed_dim, node_in_dim[1])

        # if seq_in and seq_embed_mode=="PhysicsPCA":
        #    node_in_dim = (node_in_dim[0] + seq_embed_dim, node_in_dim[1])
        # if seq_in and seq_embed_mode=="ESM2":
        #    node_in_dim = (node_in_dim[0] + seq_embed_dim, node_in_dim[1])

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )
        # GVPConvLayer is a nn.Module that forms messages using a GVPConv and updates the node embeddings as described in the paper
        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

        self.dense = nn.Sequential(
            nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2 * ns, 1)
        )

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None):
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        # print(seq)
        if seq is not None:
            if len(seq.shape) == 1:
                seq = self.W_s(seq)
                h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
            else:  # seq representation from ESM2, PhysicsPCA or Atchleyfactor
                h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)  # in gearnet added h_edge = self.rbf((pos_out - pos_in).norm(dim=-1)), vec_edge why?
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        # if batch is None: out = out.mean(dim=0, keepdims=True)
        # else: out = scatter_mean(out, batch, dim=0)

        return out


#### this can be move to the outside models.py
class GVPEncoder(nn.Module):  # embedding table can be tuned
    def __init__(self, configs=None):
        super(GVPEncoder, self).__init__()
        node_in_dim = [6, 3]  # default
        if configs.model.struct_encoder.use_foldseek:
            node_in_dim[0] += 10  # foldseek has 10 more node scalar features

        if configs.model.struct_encoder.use_foldseek_vector:
            node_in_dim[1] += 6  # foldseek_vector has 6 more node vector features

        node_in_dim = tuple(node_in_dim)
        if configs.model.struct_encoder.use_rotary_embeddings:
            if configs.model.struct_encoder.rotary_mode==3:
                edge_in_dim = (configs.model.struct_encoder.num_rbf+8,1) #16+2+3+3 only for mode ==3 add 8D pos_embeddings
            else: 
                edge_in_dim = (configs.model.struct_encoder.num_rbf+2,1) #16+2 
        else:
            edge_in_dim = (
            configs.model.struct_encoder.num_rbf + configs.model.struct_encoder.num_positional_embeddings,
            1)  # num_rbf+num_positional_embeddings

        node_h_dim = configs.model.struct_encoder.node_h_dim
        # node_h_dim=(100, 16) #default
        # node_h_dim = (100, 32)  # seems best?
        edge_h_dim = configs.model.struct_encoder.edge_h_dim
        # edge_h_dim = (32, 1) #default
        gvp_num_layers = configs.model.struct_encoder.gvp_num_layers
        # gvp_num_layers = 3

        self.use_seq = configs.model.struct_encoder.use_seq.enable
        if self.use_seq:
            self.seq_embed_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
            self.backbone = structure_encoder(node_in_dim, node_h_dim,
                                              edge_in_dim, edge_h_dim, seq_in=True,
                                              seq_embed_mode=self.seq_embed_mode,
                                              seq_embed_dim=configs.model.struct_encoder.use_seq.seq_embed_dim,
                                              num_layers=gvp_num_layers)
        else:
            self.backbone = structure_encoder(node_in_dim, node_h_dim,
                                              edge_in_dim, edge_h_dim, seq_in=False,
                                              num_layers=gvp_num_layers)

        # pretrained_state_dict = init_model.state_dict()
        # self.backbone.load_state_dict(pretrained_state_dict,strict=False)

    def forward(self, graph,
                esm2_representation=None):  # this batch is torch_geometric batch.batch to indicate the batch
        """
        graph: torch_geometric batchdasta
        """
        nodes = (graph.node_s, graph.node_v)
        edges = (graph.edge_s, graph.edge_v)
        if self.use_seq and self.seq_embed_mode != "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges,
                                                      seq=graph.seq)
        elif self.use_seq and self.seq_embed_mode == "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges,
                                                      seq=esm2_representation
                                                      )
        else:
            residue_feature_embedding = self.backbone(nodes, graph.edge_index, edges,
                                                      seq=None)

        # graph_feature=scatter_mean(residue_feature, batch, dim=0)
        graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph.batch)
        return graph_feature_embedding, residue_feature_embedding
