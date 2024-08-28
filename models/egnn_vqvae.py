import torch.nn as nn
import torch
import numpy as np
from vector_quantize_pytorch import VectorQuantize
from utils.utils import print_trainable_parameters
from egnn_pytorch import EGNN_Network
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ConvNeXtBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, sequence_length):
        super(ConvNeXtBlock, self).__init__()
        self.dw_conv = nn.Conv1d(input_dim, input_dim, kernel_size=7, padding=3,
                                 groups=input_dim)  # Depthwise convolution
        self.norm = nn.LayerNorm([sequence_length, input_dim])
        self.pw_conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)  # Pointwise convolution
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)  # Pointwise convolution

    def forward(self, x):
        residual = x
        out = self.dw_conv(x)
        out = out.permute(0, 2, 1)  # Change to (batch, sequence, channels) for LayerNorm
        out = self.norm(out)
        out = out.permute(0, 2, 1)  # Change back to (batch, channels, sequence)
        out = self.pw_conv1(out)
        out = self.gelu(out)
        out = self.pw_conv2(out)
        out += residual
        return out


class EGNNVQVAE3DTransformer(nn.Module):
    def __init__(self, latent_dim, codebook_size, decay, configs):
        super(EGNNVQVAE3DTransformer, self).__init__()

        self.max_length = configs.model.max_length

        # Define the number of residual blocks for encoder and decoder
        self.num_encoder_blocks = configs.model.vqvae.encoder.num_blocks
        self.num_decoder_blocks = configs.model.vqvae.decoder.num_blocks
        self.encoder_dim = configs.model.vqvae.encoder.dimension
        self.decoder_dim = configs.model.vqvae.decoder.dimension

        # self.embedding_encoder = nn.Embedding(num_embeddings=self.max_length+1, embedding_dim=128, padding_idx=0)

        self.egnn_model = EGNN_Network(
            num_tokens=self.max_length + 1,
            num_positions=self.max_length,
            # unless what you are passing in is an unordered set, set this to the maximum sequence length
            dim=128,
            # m_dim=64,
            depth=2,
            # global_linear_attn_every=1,
            # global_linear_attn_heads=8,
            # global_linear_attn_dim_head=32,
            # norm_feats=True,  # whether to layernorm the features
            norm_coors=True,
            # whether to normalize the coordinates, using a strategy from the SE(3) Transformers paper
            # num_nearest_neighbors=4,
            # coor_weights_clamp_value=10.0,
            # absolute clamped value for the coordinate weights, needed if you increase the num neareest neighbors
        )
        input_shape = 128
        # Encoder
        self.encoder_tail = nn.Sequential(
            nn.Conv1d(input_shape, self.encoder_dim, kernel_size=1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim, nhead=8, dim_feedforward=self.encoder_dim * 4, activation='gelu', dropout=0.0,
            batch_first=True
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_blocks)

        self.pos_embed_encoder = nn.Parameter(torch.randn(1, self.max_length, latent_dim) * .02)

        self.encoder_head = nn.Sequential(
            nn.Conv1d(self.encoder_dim, latent_dim, 1),
        )

        # Vector Quantizer layer
        # self.vector_quantizer = VectorQuantize(
        #     dim=latent_dim,
        #     codebook_size=codebook_size,
        #     decay=decay,
        #     commitment_weight=configs.model.vqvae.vector_quantization.commitment_weight,
        #     # orthogonal_reg_weight=10,  # in paper, they recommended a value of 10
        #     # orthogonal_reg_max_codes=512,
        #     # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
        #     # orthogonal_reg_active_codes_only=False
        #     # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
        # )

        self.pos_embed_decoder = nn.Parameter(torch.randn(1, self.max_length, latent_dim) * .02)

        self.decoder_tail = nn.Sequential(
            nn.Conv1d(latent_dim, self.decoder_dim, 1),
        )

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_dim, nhead=8, dim_feedforward=self.decoder_dim * 4, activation='gelu', dropout=0.0,
            batch_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=self.num_decoder_blocks)

        # todo: build GCPnet here
        # self.decoder_head = GCPNet()

        self.decoder_head = nn.Sequential(
            ResidualBlock(self.decoder_dim, self.decoder_dim),
            nn.Conv1d(self.decoder_dim, 9, 1),
        )

    def forward(self, batch, return_vq_only=False):
        initial_x = batch['input_coords']
        mask = batch['masks']
        feats = torch.tensor(range(1, self.max_length + 1)).reshape(1, self.max_length).to(initial_x.device)
        feats = feats.repeat_interleave(mask.shape[0], dim=0)
        x = self.egnn_model(feats, initial_x, mask)

        x = x[0]

        x = x.permute(0, 2, 1)
        x = self.encoder_tail(x)

        x = x.permute(0, 2, 1)
        # Apply positional encoding to encoder
        x = x + self.pos_embed_encoder
        x = self.encoder_blocks(x)
        x = x.permute(0, 2, 1)

        x = self.encoder_head(x)

        # x = x.permute(0, 2, 1)
        # x, indices, commit_loss = self.vector_quantizer(x)
        # x = x.permute(0, 2, 1)

        # if return_vq_only:
        #     return x, indices, commit_loss

        # Apply positional encoding to decoder
        x = self.decoder_tail(x)

        x = x.permute(0, 2, 1)
        x = x + self.pos_embed_decoder
        x = self.decoder_blocks(x)

        # todo: use GCPnet here
        # x shape is (batch, number of amino acids, decoder dim) e.g., (32, 128, 256)
        # mask shape is (batch, number of amino acids) e.g., (32, 128)
        # x = self.decoder_head(x, mask)

        # return x, indices, commit_loss
        return x, torch.Tensor([0]).to(x.device), torch.Tensor([0]).to(x.device)  # dummy return values for vq layer


def prepare_models_vqvae(configs, logger, accelerator):
    vqvae = EGNNVQVAE3DTransformer(
        latent_dim=configs.model.vqvae.vector_quantization.dim,
        codebook_size=configs.model.vqvae.vector_quantization.codebook_size,
        decay=configs.model.vqvae.vector_quantization.decay,
        configs=configs
    )
    if accelerator.is_main_process:
        print_trainable_parameters(vqvae, logger, 'VQ-VAE')

    return vqvae


if __name__ == '__main__':
    import yaml
    import tqdm
    from utils.utils import load_configs, get_dummy_logger
    from torch.utils.data import DataLoader
    from accelerate import Accelerator
    from data.dataset import EGNNVQVAEDataset

    config_path = "../configs/config_egnn_vqvae.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    test_logger = get_dummy_logger()
    accelerator = Accelerator()

    test_model = prepare_models_vqvae(test_configs, test_logger, accelerator)

    dataset = EGNNVQVAEDataset(test_configs.valid_settings.data_path,
                               train_mode=False, rotate_randomly=False,
                               max_samples=test_configs.train_settings.max_task_samples,
                               configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=test_configs.train_settings.batch_size,
                             num_workers=test_configs.train_settings.num_workers, pin_memory=True)
    struct_embeddings = []
    test_model.eval()
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        output, _, _ = test_model(batch)
        # print(output.shape)
        # break
