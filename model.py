import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from vector_quantize_pytorch import VectorQuantize, LFQ
import gvp.models
from torch_geometric.nn import radius, global_mean_pool, global_max_pool
from data import *


class TransformersVQAutoEncoder(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_encoder_layers=6, dim_feedforward=64, **kwargs):
        super().__init__()

        self.d_model = kwargs['dim']
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward

        # Initial Convolution Block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Transformer Encoder Block
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, activation='gelu', batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.positional_encoding = nn.Parameter(torch.randn(1, 196, self.d_model))

        # Vector Quantization Layer as before
        self.vq_layer = VectorQuantize(
            dim=self.d_model,
            codebook_size=kwargs['codebook_size'],
            decay=kwargs['decay'],
            commitment_weight=kwargs['commitment_weight'],
            accept_image_fmap=True
        )

        # Decoder Convolution Block
        self.decoder_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.Conv2d(self.d_model, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),

        ])

    def forward(self, x, return_vq_only=False):
        x = self.initial_conv(x)
        n, c, h, w = x.shape
        x = x.view(n, h * w, c)  # Adjust to (batch_size, seq_length, feature_size)
        x += self.positional_encoding[:, :h * w, :]  # Ensure positional encoding is added correctly
        x = self.transformer_encoder(x)
        x = x.view(n, c, h, w)  # Reshape back to feature map

        x, indices, commit_loss = self.vq_layer(x)

        if return_vq_only:
            return x, indices, commit_loss

        for layer in self.decoder_layers:
            x = layer(x)

        return x.clamp(-1, 1), indices, commit_loss


class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.d_model = kwargs['dim']

        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            # nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        self.vq_layer = VectorQuantize(
            dim=self.d_model,
            codebook_size=kwargs['codebook_size'],
            decay=kwargs['decay'],
            commitment_weight=kwargs['commitment_weight'],
            accept_image_fmap=True
        )
        # from math import log2
        # self.vq_layer = LFQ(
        #     dim=self.d_model,
        #     codebook_size=kwargs['codebook_size'],
        #     entropy_loss_weight=0.02,
        #     diversity_gamma=1.0
        # )

        self.decoder_layers = nn.ModuleList([
            # nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv2d(self.d_model, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x, return_vq_only=False):
        for layer in self.encoder_layers:
            x = layer(x)
            # print(x.shape)

        x, indices, commit_loss = self.vq_layer(x)

        if return_vq_only:
            return x, indices, commit_loss

        for layer in self.decoder_layers:
            x = layer(x)
            # print(x.shape)
        # make sure the output is in the range of 0-1
        x = torch.clamp(x, 0, 1)
        return x, indices, commit_loss


class GVPEncoder(nn.Module):  # embedding table can be tuned
    def __init__(self, residue_inner_dim=4096,
                 residue_out_dim=256,
                 protein_out_dim=256,
                 residue_num_projector=2,
                 protein_inner_dim=4096,
                 protein_num_projector=2,
                 seqlen=512):
        super(GVPEncoder, self).__init__()
        node_in_dim = [6, 3]  # default

        node_in_dim = tuple(node_in_dim)
        edge_in_dim = (
            16 + 16,
            1)  # num_rbf+num_positional_embeddings

        node_h_dim = (100, 16)  # default
        # node_h_dim = (100, 32)  # seems best?
        edge_h_dim = (32, 1)  # default
        gvp_num_layers = 3

        self.use_seq = False
        if self.use_seq:
            self.seq_embed_mode = configs.model.struct_encoder.use_seq.seq_embed_mode
            self.backbone = structure_encoder(node_in_dim, node_h_dim,
                                              edge_in_dim, edge_h_dim, seq_in=True,
                                              seq_embed_mode=self.seq_embed_mode,
                                              seq_embed_dim=configs.model.struct_encoder.use_seq.seq_embed_dim,
                                              num_layers=gvp_num_layers)
        else:
            self.backbone = gvp.models.structure_encoder(node_in_dim, node_h_dim,
                                                         edge_in_dim, edge_h_dim, seq_in=False,
                                                         num_layers=gvp_num_layers)

        # pretrained_state_dict = init_model.state_dict()
        # self.backbone.load_state_dict(pretrained_state_dict,strict=False)

    def forward(self, graph,
                esm2_representation=None):  # this batch is torch_geometric batch.batch to indicate the batch
        """
        graph: torch_geometric batchdasta
        """
        nodes = (graph['graph'].node_s, graph['graph'].node_v)
        edges = (graph['graph'].edge_s, graph['graph'].edge_v)
        if self.use_seq and self.seq_embed_mode != "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph['graph'].edge_index, edges,
                                                      seq=graph.seq)
        elif self.use_seq and self.seq_embed_mode == "ESM2":
            residue_feature_embedding = self.backbone(nodes, graph['graph'].edge_index, edges,
                                                      seq=esm2_representation
                                                      )
        else:
            residue_feature_embedding = self.backbone(nodes, graph['graph'].edge_index, edges,
                                                      seq=None)

        # graph_feature=scatter_mean(residue_feature, batch, dim=0)
        graph_feature_embedding = global_mean_pool(residue_feature_embedding, graph['graph'].batch)
        return graph_feature_embedding, residue_feature_embedding


def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model, logging, description=""):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    logging.info(
        f"{description} trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_models():
    residue_inner_dim = 4096,
    residue_out_dim = 256,
    protein_out_dim = 256,
    residue_num_projector = 2,
    protein_inner_dim = 4096,
    protein_num_projector = 2,
    seqlen = 512
    gvp_model = GVPEncoder(residue_inner_dim=4096,
                           residue_out_dim=256,
                           protein_out_dim=256,
                           residue_num_projector=2,
                           protein_inner_dim=4096,
                           protein_num_projector=2,
                           seqlen=512)

    # model = SimpleVQAutoEncoder(
    #     dim=configs.model.vector_quantization.dim,
    #     codebook_size=configs.model.vector_quantization.codebook_size,
    #     decay=configs.model.vector_quantization.decay,
    #     commitment_weight=configs.model.vector_quantization.commitment_weight
    # )

    # if accelerator.is_main_process:
    #     print_trainable_parameters(model, logging, 'VQ-VAE')

    return gvp_model


if __name__ == '__main__':
    import yaml
    # from utils import get_dummy_logger
    from utils import load_configs

    # logger, buffer = get_dummy_logger()

    config_path = "./config.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    main_configs = load_configs(config_file)

    net = SimpleVQAutoEncoder(
        dim=main_configs.model.vector_quantization.dim,
        codebook_size=main_configs.model.vector_quantization.codebook_size,
        decay=main_configs.model.vector_quantization.decay,
        commitment_weight=main_configs.model.vector_quantization.commitment_weight
    )

    dataset_path = './data/h5' # test for piece of data
    dataset = ProteinGraphDataset(dataset_path)

    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)

    for batch in test_dataloader:
        model = prepare_models()
        output = model(batch)
        print(output)

    # create a random input tensor and pass it through the network
    # x = torch.randn(1, 3, 32, 32)
    # output, x, y = net(x, return_vq_only=False)
    # print(output.shape)
    # print(x.shape)
    # print(y.shape)
