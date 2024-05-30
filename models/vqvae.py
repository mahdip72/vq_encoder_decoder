import torch.nn as nn
import torch
from vector_quantize_pytorch import VectorQuantize
from utils.utils import print_trainable_parameters

import torch
import torch.nn as nn
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


class VQVAE3DResNet(nn.Module):
    def __init__(self, input_dim, latent_dim, codebook_size, decay, configs):
        super(VQVAE3DResNet, self).__init__()

        self.max_length = configs.model.max_length

        # Define the number of residual blocks for encoder and decoder
        self.num_encoder_blocks = 8
        self.num_decoder_blocks = 8

        # Encoder
        self.initial_conv = nn.Sequential(
            nn.Conv1d(12, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU()
        )

        self.encoder_blocks = nn.Sequential(
            *[ResidualBlock(input_dim, input_dim) for _ in range(self.num_encoder_blocks)],
            nn.Conv1d(input_dim, latent_dim, 3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=1.0,
        )

        # Decoder
        self.decoder_blocks = nn.Sequential(
            nn.Conv1d(latent_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            *[ResidualBlock(input_dim, input_dim) for _ in range(self.num_decoder_blocks)]
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(input_dim, 12, 1),
        )

    def forward(self, batch, return_vq_only=False):
        x = batch['coords'].permute(0, 2, 1)

        x = self.initial_conv(x)
        x = self.encoder_blocks(x)

        x = x.permute(0, 2, 1)
        x, indices, commit_loss = self.vector_quantizer(x)
        x = x.permute(0, 2, 1)

        if return_vq_only:
            return x, indices, commit_loss

        x = self.decoder_blocks(x)
        x = self.final_conv(x)

        x = x.permute(0, 2, 1)
        return x, indices, commit_loss


class VQVAE3D(nn.Module):
    def __init__(self, input_dim, latent_dim, codebook_size, decay, configs):
        super(VQVAE3D, self).__init__()

        self.max_length = configs.model.max_length

        self.encoder_layers = nn.Sequential(
            nn.Conv1d(12, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, latent_dim, 3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

        self.vector_quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=1.0,
            # accept_image_fmap=True,
        )
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(latent_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),

            nn.Conv1d(input_dim, input_dim, 1),
            nn.BatchNorm1d(input_dim),
        )

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, 12, 1),
            # nn.Tanh()
        )

    def forward(self, batch, return_vq_only=False):
        # change the shape of x from (batch, num_nodes, node_dim) to (batch, node_dim, num_nodes)
        x = batch['coords'].permute(0, 2, 1)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.permute(0, 2, 1)
        x, indices, commit_loss = self.vector_quantizer(x)
        x = x.permute(0, 2, 1)

        if return_vq_only:
            return x, indices, commit_loss

        for layer in self.decoder_layers:
            x = layer(x)

        for layer in self.head:
            x = layer(x)

        # make it to be (batch, num_nodes, 12)
        x = x.permute(0, 2, 1)
        return x, indices, commit_loss


class VQVAE3DTransformer(nn.Module):
    def __init__(self, codebook_size, decay, configs):
        super(VQVAE3DTransformer, self).__init__()

        self.max_length = configs.model.max_length

        # Projecting the input to the dimension expected by the Transformer
        self.input_projection = nn.Linear(12, configs.model.vqvae.encoder.dimension)

        self.pos_embed = nn.Parameter(torch.randn(1, self.max_length, configs.model.vqvae.encoder.dimension) * .02)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.model.vqvae.encoder.dimension,
            nhead=configs.model.vqvae.encoder.num_heads,
            dim_feedforward=configs.model.vqvae.encoder.dim_feedforward,
            activation=configs.model.vqvae.encoder.activation_function
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=configs.model.vqvae.encoder.num_layers)

        # Projecting the output of the Transformer to the dimension expected by the VQ layer
        self.vq_in_projection = nn.Linear(configs.model.vqvae.encoder.dimension, configs.model.vqvae.vector_quantization.dim)

        # Vector Quantizer
        self.vector_quantizer = VectorQuantize(
            dim=configs.model.vqvae.vector_quantization.dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=1.0,
        )

        # Projecting the output of the VQ layer back to the decoder dimension
        self.vq_out_projection = nn.Linear(configs.model.vqvae.vector_quantization.dim, configs.model.vqvae.encoder.dimension)

        # Transformer Decoder
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.model.vqvae.decoder.dimension,
            nhead=configs.model.vqvae.decoder.num_heads,
            dim_feedforward=configs.model.vqvae.decoder.dim_feedforward,
            activation=configs.model.vqvae.encoder.activation_function
        )
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=configs.model.vqvae.decoder.num_layers)

        # Projecting the output back to the original dimension
        self.output_projection = nn.Linear(configs.model.vqvae.encoder.dimension, 12)

    def drop_positional_encoding(self, embedding):
        embedding = embedding + self.pos_embed
        return embedding

    def forward(self, batch, return_vq_only=False):
        x = batch['coords']

        # Apply input projection
        x = self.input_projection(x)

        # Apply positional encoding
        x = self.drop_positional_encoding(x)

        # Permute for Transformer [batch, sequence, feature]
        x = x.permute(1, 0, 2)

        # Encoder
        x = self.encoder(x)

        # Apply qv_in_projection
        x = self.vq_in_projection(x)

        # Permute back to [batch, feature, sequence]
        x = x.permute(1, 0, 2)
        x, indices, commit_loss = self.vector_quantizer(x)

        if return_vq_only:
            return x, indices, commit_loss

        # Apply vq_out_projection
        x = self.vq_out_projection(x)

        # Permute for Transformer Decoder [sequence, batch, feature]
        x = x.permute(1, 0, 2)

        # Decoder
        x = self.decoder(x)

        # Permute back to [batch, sequence, feature]
        x = x.permute(1, 0, 2)

        # Apply output projection
        x = self.output_projection(x)

        return x, indices, commit_loss


def prepare_models_vqvae(configs, logger, accelerator):
    vqvae = VQVAE3DResNet(
        input_dim=configs.model.vqvae.vector_quantization.dim*4,
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
    from data.dataset import VQVAEDataset

    config_path = "../configs/config_gvp.yaml"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    test_configs = load_configs(config_file)

    test_logger = get_dummy_logger()
    accelerator = Accelerator()

    test_model = prepare_models_vqvae(test_configs, test_logger, accelerator)

    dataset = VQVAEDataset(test_configs.train_settings.data_path, configs=test_configs)

    test_loader = DataLoader(dataset, batch_size=test_configs.train_settings.batch_size,
                             num_workers=test_configs.train_settings.num_workers, pin_memory=True)
    struct_embeddings = []
    test_model.eval()
    for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
        output, _, _ = test_model(batch)
        print(output.shape)
        break
